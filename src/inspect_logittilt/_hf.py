"""LogitTilt on top of Inspect's HuggingFace provider.

Subclasses HuggingFaceAPI and overrides only generate(), so model loading,
tokenizer setup and chat templating are inherited -- and hf-logittilt
accepts every model_arg hf/ does. inspect_ai.model._providers.hf is
private, so tests/test_provider.py exercises the surface we depend on.
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
    """Token ids that end a turn.

    A chat model rarely emits the document EOS; it uses a turn marker whose
    position in generation_config differs by family. So render a closed
    assistant turn and keep whichever candidates the template emits.
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

    A raw model() call does not derive these from the mask, so without them a
    padded row silently gets different logits from the same text unpadded.
    """
    return (attention_mask.cumsum(-1) - 1).clamp(min=0)


def truncate_at_stop(completion: str, stop_seqs: list[str] | None) -> str:
    """Cut the completion at the earliest stop sequence, excluding it."""
    if not stop_seqs:
        return completion
    cut = min(
        (completion.index(stop) for stop in stop_seqs if stop in completion),
        default=None,
    )
    return completion if cut is None else completion[:cut]


def token_probability_summary(target_logprobs: list[float]) -> dict[str, float | int]:
    """On-policy plausibility of a completion, as percentages."""
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
# long enough to catch the burst Inspect fires off together
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
    # a snapshot: self.tilt can change between queueing and decoding
    tilt: TiltConfig
    future: asyncio.Future = field(repr=False)


class LogitTiltHFAPI(HuggingFaceAPI):
    """HuggingFace target model whose decoding is steered toward a behaviour."""

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        # mirrors HuggingFaceAPI: Inspect constructs providers positionally
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

        The prompt goes at the start as a system message, merged into an existing
        one if present -- several templates reject a system message that is not
        first. The reminder goes on the last user message, next to where
        generation begins.
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

        In an agentic loop that message sits behind the tool exchanges rather than
        next to where generation resumes, so a long tool history dilutes it. Reach
        for `prefill` instead when it does -- that lands at the generation point.
        """
        messages = list(messages)
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].role == "user" and isinstance(messages[i].content, str):
                separator = "\n\n"
                messages[i] = ChatMessageUser(content=f"{messages[i].content}{separator}{reminder}")
                return messages
        return [*messages, ChatMessageUser(content=reminder)]

    def _elicited_messages_via_user(self, input: list[ChatMessage]) -> list[ChatMessage]:
        """Fallback for templates that will not carry a system message."""
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

        Checks the instruction survived templating: some templates drop system
        content silently, which would make steering a no-op that still looks fine.
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
            # lands where the reply begins; elicited context only
            elicited = elicited + self.tilt.prefill
        return target, elicited

    # ------------------------------------------------------------------
    # decoding
    # ------------------------------------------------------------------

    def _encode_left_padded(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenise a batch with LEFT padding, since generation continues from the
        final position.
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
        """Step both contexts in lockstep for a batch.

        Two KV caches advance over the same sampled tokens, so the distributions
        differ only in their prefix. Rows stop independently; finished ones are
        fed pad to keep the batch rectangular.
        """
        if len(target_texts) != len(elicited_texts) or len(target_texts) != len(max_tokens):
            raise ValueError("target, elicited and max_tokens batches must be the same length")
        batch_size = len(target_texts)

        # always generated: it carries the floor threshold and the reported
        # probability, not just the mixture
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

        # reproducible for a fixed batch; composition still varies with timing
        generator = None
        if seed is not None:
            generator = torch.Generator(device=device)
            generator.manual_seed(int(seed))

        # decoded fresh each step so token boundaries cannot split a match
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

            # finished rows are fed pad to keep the batch rectangular
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

        An asyncio.Queue binds to its creating loop, so both are rebuilt when the
        loop changes.
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

            # a batch shares one sampling rule; the rest goes back on the queue
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
        """Per-token logprobs from the UNMODIFIED target.

        Upstream reports the distribution it sampled from; the tokens here were
        chosen under the tilt while the probabilities describe what the base model
        thought of them.
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

        # hf_chat already put the tools in both prompts; the upstream handler
        # parses any call back out of the completion
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
