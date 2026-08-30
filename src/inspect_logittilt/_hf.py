"""LogitTilt on top of Inspect's HuggingFace provider.

``LogitTiltHFAPI`` subclasses ``HuggingFaceAPI`` rather than reimplementing it.
Model loading, tokenizer setup and chat templating are all inherited; the only
method we override is ``generate()``, because that is the only thing LogitTilt
changes. The practical consequence for users is that ``hf-logittilt`` accepts
every ``model_args`` that ``hf`` accepts -- ``device``, ``tokenizer_path``,
``trust_remote_code``, and so on -- plus the LogitTilt ones.

Note that ``inspect_ai.model._providers.hf`` is a private module. We depend on a
deliberately small part of it (``__init__``, ``self.model``, ``self.tokenizer``,
``hf_chat``); ``tests/test_provider.py`` exercises that surface so an upstream
change breaks loudly in CI rather than silently for users.
"""

from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import dataclass, field
from typing import Any

import torch
from inspect_ai.model import (
    ChatCompletionChoice,
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageUser,
    GenerateConfig,
    Logprob,
    Logprobs,
    ModelOutput,
    TopLogprob,
)
from inspect_ai.model._providers.hf import HuggingFaceAPI
from inspect_ai.model._providers.util import ChatAPIHandler, HFHandler
from inspect_ai.tool import ToolChoice, ToolInfo

from ._tilt import TiltConfig, build_config, sample_next

logger = logging.getLogger(__name__)


def stop_token_ids(tokenizer: Any, model: Any) -> set[int]:
    """Token ids that should end generation.

    A chat model rarely emits the document EOS (``tokenizer.eos_token_id``); it
    ends a turn with a turn marker instead -- ``<|im_end|>`` for Qwen,
    ``<end_of_turn>`` for Gemma, ``<|end|>`` for Phi. ``generation_config`` often
    lists several candidates and their order differs between model families, so
    we cannot just take the first. Instead we render a complete assistant turn
    and keep whichever candidates the chat template actually emits to close it.
    """
    candidates: set[int] = set()

    generation_config = getattr(model, "generation_config", None)
    configured = getattr(generation_config, "eos_token_id", None)
    if isinstance(configured, int):
        candidates.add(configured)
    elif isinstance(configured, (list, tuple)):
        candidates.update(int(token_id) for token_id in configured)

    if tokenizer.eos_token_id is not None:
        candidates.add(int(tokenizer.eos_token_id))

    try:
        closed_turn = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "answer"},
            ],
            tokenize=True,
            add_generation_prompt=False,
        )
        emitted = candidates & {int(token_id) for token_id in closed_turn}
        if emitted:
            return emitted
    except Exception as exc:  # noqa: BLE001 - not every tokenizer has a chat template
        logger.debug("could not narrow stop tokens via the chat template: %s", exc)

    return candidates


def positions_from_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    """Position ids for a LEFT-padded batch.

    A raw ``model(...)`` call does not derive positions from the mask -- only
    ``generate()``'s input preparation does that. Left padding shifts real tokens
    to later absolute positions, so without explicit position ids a padded row
    silently gets different positions, and therefore different logits, from the
    same text unpadded. Pad slots are clamped to 0; they are masked out anyway.
    """
    return (attention_mask.cumsum(-1) - 1).clamp(min=0)


def truncate_at_stop(completion: str, stop_seqs: list[str] | None) -> str:
    """Cut the completion at the earliest stop sequence, excluding it.

    Decoding stops a row as soon as a stop sequence appears in its tail, but the
    token that completed the match usually carries trailing text as well, so the
    string still has to be trimmed. Excluding the sequence itself matches what
    every other provider returns.
    """
    if not stop_seqs:
        return completion
    cut = min(
        (completion.index(stop) for stop in stop_seqs if stop in completion),
        default=None,
    )
    return completion if cut is None else completion[:cut]


def token_probability_summary(target_logprobs: list[float]) -> dict[str, float | int]:
    """On-policy plausibility of a completion, as percentages.

    These come free from the target logits already computed at each step, and are
    the metrics the method is tuned against: how probable the *unmodified* model
    considers the text that steering produced.
    """
    if not target_logprobs:
        return {"tokens": 0}
    probs = [math.exp(lp) for lp in target_logprobs]
    return {
        "tokens": len(probs),
        "arithmetic_mean_token_prob": 100.0 * sum(probs) / len(probs),
        "geometric_mean_token_prob": 100.0 * math.exp(sum(target_logprobs) / len(probs)),
        "min_token_prob": 100.0 * min(probs),
    }


DEFAULT_BATCH_SIZE = 8
# how long to keep collecting arrivals after the first one. Inspect fires its
# samples off together, so a few milliseconds is enough to catch the burst, and
# it is nothing against a multi-second generation.
BATCH_LINGER_SECONDS = 0.01


@dataclass
class _PendingRequest:
    """One generate() call waiting for a batch to form."""

    target_text: str
    elicited_text: str
    max_tokens: int
    temperature: float
    top_k: int | None
    top_p: float | None
    seed: int | None
    stop_seqs: list[str] | None
    top_logprobs: int | None
    # snapshot, not a reference to self.tilt: the provider's config can be
    # mutated between queueing and decoding, and reporting settings that were
    # not the ones actually used is worse than not reporting them.
    tilt: TiltConfig
    future: asyncio.Future = field(repr=False)


class LogitTiltHFAPI(HuggingFaceAPI):
    """HuggingFace target model whose decoding is steered toward a named behaviour."""

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        # signature mirrors HuggingFaceAPI/ModelAPI exactly; Inspect constructs
        # providers positionally, so it must not diverge.
        config: GenerateConfig = GenerateConfig(),  # noqa: B008
        **model_args: Any,
    ) -> None:
        self.tilt, passthrough = build_config(model_args)
        super().__init__(
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            config=config,
            **passthrough,
        )
        self._stop_ids = stop_token_ids(self.tokenizer, self.model)

    # ------------------------------------------------------------------
    # context construction
    # ------------------------------------------------------------------

    def _elicited_messages(self, input: list[ChatMessage]) -> list[ChatMessage]:
        """Conversation with the steering instruction attached.

        ``steering_prompt`` goes at the START, as a system message. If the
        conversation already opens with one -- which many tasks do, to carry
        few-shot examples or format rules -- it is merged into that rather than
        prepended as a second, because several chat templates (Qwen's among them)
        raise "System message must be at the beginning" for a system message that
        is not first.

        ``steering_reminder`` goes at the END, appended to the FINAL user
        message. Last rather than first: the point of it is to sit next to where
        generation begins, and in a multi-turn conversation the first user
        message is no closer than the system prompt. Because the elicited context
        is rebuilt from the real transcript on every call, the reminder re-lands
        adjacent to generation each turn without accumulating.
        """
        messages = list(input)

        if self.tilt.steering_prompt:
            head = messages[0] if messages else None
            if head is not None and head.role == "system" and isinstance(head.content, str):
                separator = "\n\n"
                messages = [
                    ChatMessageSystem(
                        content=f"{self.tilt.steering_prompt}{separator}{head.content}"
                    ),
                    *messages[1:],
                ]
            else:
                messages = [ChatMessageSystem(content=self.tilt.steering_prompt), *messages]

        if self.tilt.steering_reminder:
            messages = self._append_reminder(messages, self.tilt.steering_reminder)

        return messages

    @staticmethod
    def _append_reminder(messages: list[ChatMessage], reminder: str) -> list[ChatMessage]:
        """Append the reminder to the last user message.

        If the conversation has no user message at all -- or ends with tool
        results several messages after the last user turn, as an agentic loop
        can -- we fall back to a trailing user message. That keeps the reminder
        adjacent to generation, which is the whole point, though we have no
        measurement of how well it works in the tool case.
        """
        messages = list(messages)
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].role == "user" and isinstance(messages[i].content, str):
                separator = "\n\n"
                messages[i] = ChatMessageUser(content=f"{messages[i].content}{separator}{reminder}")
                return messages
        return [*messages, ChatMessageUser(content=reminder)]

    def _elicited_messages_via_user(self, input: list[ChatMessage]) -> list[ChatMessage]:
        """Fallback when a chat template will not carry a system message.

        Every chat template supports a user turn; not all support a system one,
        and some silently drop system content instead of raising. The steering
        prompt is prepended to the first user message; any reminder is still
        appended to the last, as usual.
        """
        separator = "\n\n"
        messages = list(input)
        prompt = self.tilt.steering_prompt

        if prompt:
            for i, message in enumerate(messages):
                if message.role == "user" and isinstance(message.content, str):
                    messages[i] = ChatMessageUser(content=f"{prompt}{separator}{message.content}")
                    break
            else:
                messages = [ChatMessageUser(content=prompt), *messages]

        if self.tilt.steering_reminder:
            messages = self._append_reminder(messages, self.tilt.steering_reminder)
        return messages

    def _contexts(self, input: list[ChatMessage], tools: list[ToolInfo]) -> tuple[str, str]:
        """Render the two prompts the tilt runs over.

        The target context is the conversation as it stands. The elicited context
        is the same conversation plus the steering instruction. Both go through
        the inherited ``hf_chat()``, so they use the model's real chat template
        rather than anything we invent.

        The instruction is attached as a system message by default, matching the
        paper. Some chat templates do not support a system role and drop it
        silently, which would turn steering into a no-op without any error, so we
        check that the instruction survives rendering and fall back to the first
        user message if it did not. The check is behavioural: no model-name
        special cases.
        """
        target = self.hf_chat(input, tools)

        elicited = self.hf_chat(self._elicited_messages(input), tools)
        marker = self.tilt.steering_prompt or self.tilt.steering_reminder
        if marker not in elicited:
            logger.info(
                "this chat template does not carry a system message; attaching the "
                "steering instruction to the user message instead"
            )
            elicited = self.hf_chat(self._elicited_messages_via_user(input), tools)
            if marker not in elicited:
                raise RuntimeError(
                    "the steering prompt did not survive chat templating as either a "
                    "system or a user message, so steering would silently do nothing. "
                    "Inspect the model's chat template."
                )

        if self.tilt.prefill:
            # hf_chat ends with the generation prompt, so the prefill lands exactly
            # where the assistant's reply begins. It shapes the elicited
            # distribution only and never enters the returned completion.
            elicited = elicited + self.tilt.prefill
        return target, elicited

    # ------------------------------------------------------------------
    # decoding
    # ------------------------------------------------------------------

    def _encode_left_padded(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenise a batch with LEFT padding.

        Left rather than right because generation continues from the final
        position: with right padding the last token of a short row would be pad,
        and its next-token distribution meaningless.
        """
        original_side = self.tokenizer.padding_side
        try:
            self.tokenizer.padding_side = "left"
            encoded = self.tokenizer(texts, return_tensors="pt", padding=True)
        finally:
            self.tokenizer.padding_side = original_side
        device = self.model.device
        return encoded.input_ids.to(device), encoded.attention_mask.to(device)

    @torch.inference_mode()
    def _decode(
        self,
        target_texts: list[str],
        elicited_texts: list[str],
        max_tokens: list[int],
        temperature: float,
        tilt: TiltConfig,
        top_k: int | None = None,
        top_p: float | None = None,
        seed: int | None = None,
        stop_seqs: list[str] | None = None,
        top_logprobs: int | None = None,
    ) -> list[tuple[list[int], list[float], list[list[tuple[int, float]]]]]:
        """Step both contexts in lockstep for a batch of requests.

        Two KV caches advance together over the *same* sampled tokens: whatever is
        drawn is appended to both the target and the elicited context, so the two
        distributions stay conditioned on an identical continuation and differ
        only in their prefix.

        Rows are independent -- each stops at its own stop token -- but they share
        the forward passes, which is where the speedup comes from. A finished row
        keeps being fed pad tokens so shapes stay rectangular; its outputs are
        discarded.
        """
        if len(target_texts) != len(elicited_texts) or len(target_texts) != len(max_tokens):
            raise ValueError("target, elicited and max_tokens batches must be the same length")
        batch_size = len(target_texts)

        # The target stream is always generated. It has three consumers -- the
        # mixture, the naturalness floor (which thresholds on the TRUE target
        # distribution) and the reported on-policy probability -- and the last of
        # those applies to every run, so skipping it at target_strength=0 would
        # silently drop the plausibility metric. Where the stream is needed
        # anyway, capturing logprobs during decoding is free: the logits were
        # computed for the mixture regardless. (If prompted-only ever becomes a
        # long-running workflow rather than a diagnostic, the cheaper answer is
        # one teacher-forced scoring pass at the end -- one parallel forward
        # instead of T sequential steps -- with batch-chunking to keep the
        # [B, T, V] logits in memory.)
        need_target = True
        need_elicited = tilt.steering_strength != 0.0

        target_past = elicited_past = None
        target_mask = elicited_mask = None
        target_logits = elicited_logits = None

        if need_target:
            target_ids, target_mask = self._encode_left_padded(target_texts)
            target_out = self.model(
                input_ids=target_ids,
                attention_mask=target_mask,
                position_ids=positions_from_mask(target_mask),
                use_cache=True,
            )
            target_past = target_out.past_key_values
            target_logits = target_out.logits[:, -1, :].float()

        if need_elicited:
            elicited_ids, elicited_mask = self._encode_left_padded(elicited_texts)
            elicited_out = self.model(
                input_ids=elicited_ids,
                attention_mask=elicited_mask,
                position_ids=positions_from_mask(elicited_mask),
                use_cache=True,
            )
            elicited_past = elicited_out.past_key_values
            elicited_logits = elicited_out.logits[:, -1, :].float()

        reference = target_logits if target_logits is not None else elicited_logits
        device = reference.device

        # A seeded generator makes a decode reproducible for a fixed batch. Batch
        # composition itself depends on arrival timing, so identical eval runs can
        # still group requests differently -- documented rather than pretended away.
        generator = None
        if seed is not None:
            generator = torch.Generator(device=device)
            generator.manual_seed(int(seed))

        # how far back to look for a stop sequence: enough tokens to contain the
        # longest one, decoded fresh each step rather than accumulated, so token
        # boundaries cannot split a match
        stop_window = 0
        if stop_seqs:
            stop_window = max(8, max(len(s) for s in stop_seqs))
        pad_id = self.tokenizer.pad_token_id or 0
        tokens: list[list[int]] = [[] for _ in range(batch_size)]
        target_logprobs: list[list[float]] = [[] for _ in range(batch_size)]
        alternatives: list[list[list[tuple[int, float]]]] = [[] for _ in range(batch_size)]
        done = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max(max_tokens)):
            sampled, logprobs, step_alternatives = sample_next(
                target_logits,
                elicited_logits,
                tilt,
                temperature=temperature,
                generator=generator,
                top_k=top_k,
                top_p=top_p,
                top_logprobs=top_logprobs,
            )

            for row in range(batch_size):
                if done[row]:
                    continue
                token_id = int(sampled[row].item())
                if token_id in self._stop_ids:
                    done[row] = True
                    continue
                tokens[row].append(token_id)
                if logprobs is not None:
                    target_logprobs[row].append(float(logprobs[row].item()))
                if step_alternatives is not None:
                    ids, values = step_alternatives
                    alternatives[row].append(
                        [
                            (int(i), float(v))
                            for i, v in zip(ids[row].tolist(), values[row].tolist())
                        ]
                    )
                if len(tokens[row]) >= max_tokens[row]:
                    done[row] = True
                elif stop_seqs:
                    tail = self.tokenizer.decode(
                        tokens[row][-stop_window:], skip_special_tokens=True
                    )
                    if any(stop in tail for stop in stop_seqs):
                        done[row] = True

            if bool(done.all()):
                break

            # finished rows are fed pad so the batch stays rectangular
            next_input = torch.where(done, torch.full_like(sampled, pad_id), sampled)
            next_input = next_input.unsqueeze(-1)

            if need_target:
                ones = torch.ones(batch_size, 1, dtype=target_mask.dtype, device=device)
                target_mask = torch.cat([target_mask, ones], dim=-1)
                target_out = self.model(
                    input_ids=next_input,
                    attention_mask=target_mask,
                    position_ids=target_mask.sum(-1, keepdim=True) - 1,
                    past_key_values=target_past,
                    use_cache=True,
                )
                target_past = target_out.past_key_values
                target_logits = target_out.logits[:, -1, :].float()

            if need_elicited:
                ones = torch.ones(batch_size, 1, dtype=elicited_mask.dtype, device=device)
                elicited_mask = torch.cat([elicited_mask, ones], dim=-1)
                elicited_out = self.model(
                    input_ids=next_input,
                    attention_mask=elicited_mask,
                    position_ids=elicited_mask.sum(-1, keepdim=True) - 1,
                    past_key_values=elicited_past,
                    use_cache=True,
                )
                elicited_past = elicited_out.past_key_values
                elicited_logits = elicited_out.logits[:, -1, :].float()

        return list(zip(tokens, target_logprobs, alternatives, strict=True))

    # ------------------------------------------------------------------
    # ModelAPI
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # batching
    # ------------------------------------------------------------------

    def _max_batch_size(self) -> int:
        return int(self.batch_size or DEFAULT_BATCH_SIZE)

    async def _submit(self, request: _PendingRequest) -> tuple[list[int], list[float]]:
        """Queue a request and wait for the batch it lands in.

        An asyncio.Queue binds to the loop that first used it, so the queue and
        its batcher are rebuilt whenever we find ourselves on a different loop --
        otherwise a second asyncio.run() would fail with "bound to a different
        event loop". The provider outlives any single loop; the queue must not.
        """
        loop = asyncio.get_running_loop()
        if getattr(self, "_loop", None) is not loop:
            self._loop = loop
            self._queue: asyncio.Queue[_PendingRequest] = asyncio.Queue()
            self._batcher: asyncio.Task[None] | None = None

        if self._batcher is None or self._batcher.done():
            self._batcher = loop.create_task(self._run_batcher())

        await self._queue.put(request)
        return await request.future

    async def _run_batcher(self) -> None:
        """Group concurrent requests and run one lockstep decode per group."""
        while True:
            first = await self._queue.get()
            await asyncio.sleep(BATCH_LINGER_SECONDS)

            batch = [first]
            while len(batch) < self._max_batch_size():
                try:
                    batch.append(self._queue.get_nowait())
                except asyncio.QueueEmpty:
                    break

            # A batch shares one sampling rule, so it must be homogeneous in
            # everything that defines it -- temperature, tilt config and the
            # sampling options. Anything else goes back on the queue. In practice
            # an eval uses one setting throughout, so this rarely splits.
            def rule(request: _PendingRequest) -> tuple:
                return (
                    request.temperature,
                    request.tilt,
                    request.top_k,
                    request.top_p,
                    request.seed,
                    tuple(request.stop_seqs or ()),
                    request.top_logprobs,
                )

            head = rule(batch[0])
            grouped = [r for r in batch if rule(r) == head]
            deferred = [r for r in batch if rule(r) != head]
            batch = grouped
            temperature = batch[0].temperature
            tilt = batch[0].tilt
            for request in deferred:
                self._queue.put_nowait(request)

            try:
                results = await asyncio.get_running_loop().run_in_executor(
                    None,
                    self._decode,
                    [r.target_text for r in batch],
                    [r.elicited_text for r in batch],
                    [r.max_tokens for r in batch],
                    temperature,
                    tilt,
                    batch[0].top_k,
                    batch[0].top_p,
                    batch[0].seed,
                    batch[0].stop_seqs,
                    batch[0].top_logprobs,
                )
            except Exception as exc:  # noqa: BLE001 - propagate to every waiter
                for request in batch:
                    if not request.future.done():
                        request.future.set_exception(exc)
                continue

            for request, result in zip(batch, results, strict=True):
                if not request.future.done():
                    request.future.set_result(result)

    # ------------------------------------------------------------------
    # ModelAPI
    # ------------------------------------------------------------------

    def _build_logprobs(
        self,
        tokens: list[int],
        target_logprobs: list[float],
        alternatives: list[list[tuple[int, float]]],
    ) -> Logprobs:
        """Per-token logprobs, taken from the UNMODIFIED target distribution.

        A semantic difference from ``hf/`` worth knowing: upstream reports the
        distribution it sampled from, whereas the tokens here were chosen under
        the tilt while the probabilities describe what the base model thought of
        them. That is the more useful pairing for this method -- how plausible
        the unsteered model finds the steered text -- but it is not the same
        number ``hf/`` would return.
        """
        content: list[Logprob] = []
        for i, (token_id, logprob) in enumerate(zip(tokens, target_logprobs, strict=False)):
            top = [
                TopLogprob(
                    token=self.tokenizer.convert_ids_to_tokens(alt_id),
                    logprob=alt_logprob,
                    bytes=None,
                )
                for alt_id, alt_logprob in (alternatives[i] if i < len(alternatives) else [])
            ]
            content.append(
                Logprob(
                    token=self.tokenizer.convert_ids_to_tokens(token_id),
                    logprob=logprob,
                    bytes=None,
                    top_logprobs=top or None,
                )
            )
        return Logprobs(content=content)

    async def generate(
        self,
        input: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        target_text, elicited_text = self._contexts(input, tools)
        max_tokens = config.max_tokens or self.max_tokens() or 512
        temperature = config.temperature if config.temperature is not None else 1.0

        tilt = self.tilt
        stop_seqs = list(config.stop_seqs) if config.stop_seqs else None
        # config.logprobs asks for them at all; top_logprobs asks for alternatives
        wants_logprobs = bool(config.logprobs)
        top_logprobs = config.top_logprobs if wants_logprobs else None
        tokens, target_logprobs, alternatives = await self._submit(
            _PendingRequest(
                target_text=target_text,
                elicited_text=elicited_text,
                max_tokens=max_tokens,
                temperature=temperature,
                top_k=config.top_k,
                top_p=config.top_p,
                seed=config.seed,
                stop_seqs=stop_seqs,
                top_logprobs=top_logprobs,
                tilt=tilt,
                future=asyncio.get_running_loop().create_future(),
            )
        )
        completion = self.tokenizer.decode(tokens, skip_special_tokens=True)
        completion = truncate_at_stop(completion, stop_seqs)

        # Tool definitions already reach both prompts through the inherited
        # hf_chat(), which renders them with the model's own template. All that
        # is left is parsing any call back out of the text we generated, and the
        # upstream handler does exactly that -- it works on the completion
        # string, so owning the decode loop costs us nothing here.
        handler: ChatAPIHandler | None = (
            HFHandler(self.model_name, self.model_family()) if tools else None
        )
        message: ChatMessageAssistant = (
            handler.parse_assistant_response(completion, tools)
            if handler
            else ChatMessageAssistant(content=completion, model=self.model_name, source="generate")
        )

        output = ModelOutput(
            model=self.model_name,
            choices=[
                ChatCompletionChoice(
                    message=message,
                    logprobs=(
                        self._build_logprobs(tokens, target_logprobs, alternatives)
                        if wants_logprobs
                        else None
                    ),
                )
            ],
        )
        output.metadata = {
            "logittilt": {
                "steering_strength": tilt.steering_strength,
                "target_strength": tilt.target_strength,
                "naturalness_floor": tilt.naturalness_floor,
                **token_probability_summary(target_logprobs),
            }
        }
        return output
