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

import logging
import math
from typing import Any

import torch
from inspect_ai.model import (
    ChatMessage,
    ChatMessageSystem,
    ChatMessageUser,
    GenerateConfig,
    ModelOutput,
)
from inspect_ai.model._providers.hf import HuggingFaceAPI
from inspect_ai.tool import ToolChoice, ToolInfo

from ._tilt import build_config, sample_next

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
        """Conversation with the steering instruction attached, as a system message."""
        return [ChatMessageSystem(content=self.tilt.steering_prompt), *input]

    def _elicited_messages_via_user(self, input: list[ChatMessage]) -> list[ChatMessage]:
        """Fallback: attach the steering instruction to the first user message.

        Every chat template supports a user turn; not all support a system one,
        and some silently drop system content instead of raising.
        """
        messages = list(input)
        for i, message in enumerate(messages):
            if message.role == "user" and isinstance(message.content, str):
                separator = "\n\n"
                messages[i] = ChatMessageUser(
                    content=f"{self.tilt.steering_prompt}{separator}{message.content}"
                )
                return messages
        return [ChatMessageUser(content=self.tilt.steering_prompt), *messages]

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
        if self.tilt.steering_prompt not in elicited:
            logger.info(
                "this chat template does not carry a system message; attaching the "
                "steering instruction to the user message instead"
            )
            elicited = self.hf_chat(self._elicited_messages_via_user(input), tools)
            if self.tilt.steering_prompt not in elicited:
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
        max_tokens: int,
        temperature: float,
    ) -> list[tuple[list[int], list[float]]]:
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
        if len(target_texts) != len(elicited_texts):
            raise ValueError("target and elicited batches must be the same length")
        batch_size = len(target_texts)

        target_ids, target_mask = self._encode_left_padded(target_texts)
        elicited_ids, elicited_mask = self._encode_left_padded(elicited_texts)

        target_out = self.model(
            input_ids=target_ids,
            attention_mask=target_mask,
            position_ids=positions_from_mask(target_mask),
            use_cache=True,
        )
        elicited_out = self.model(
            input_ids=elicited_ids,
            attention_mask=elicited_mask,
            position_ids=positions_from_mask(elicited_mask),
            use_cache=True,
        )
        target_past, elicited_past = target_out.past_key_values, elicited_out.past_key_values
        target_logits = target_out.logits[:, -1, :].float()
        elicited_logits = elicited_out.logits[:, -1, :].float()

        device = target_logits.device
        pad_id = self.tokenizer.pad_token_id or 0
        tokens: list[list[int]] = [[] for _ in range(batch_size)]
        target_logprobs: list[list[float]] = [[] for _ in range(batch_size)]
        done = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_tokens):
            sampled, logprobs = sample_next(
                target_logits, elicited_logits, self.tilt, temperature=temperature
            )

            for row in range(batch_size):
                if done[row]:
                    continue
                token_id = int(sampled[row].item())
                if token_id in self._stop_ids:
                    done[row] = True
                    continue
                tokens[row].append(token_id)
                target_logprobs[row].append(float(logprobs[row].item()))

            if bool(done.all()):
                break

            # finished rows are fed pad so the batch stays rectangular
            next_input = torch.where(done, torch.full_like(sampled, pad_id), sampled)
            next_input = next_input.unsqueeze(-1)
            ones = torch.ones(batch_size, 1, dtype=target_mask.dtype, device=device)
            target_mask = torch.cat([target_mask, ones], dim=-1)
            elicited_mask = torch.cat([elicited_mask, ones], dim=-1)

            target_out = self.model(
                input_ids=next_input,
                attention_mask=target_mask,
                position_ids=target_mask.sum(-1, keepdim=True) - 1,
                past_key_values=target_past,
                use_cache=True,
            )
            elicited_out = self.model(
                input_ids=next_input,
                attention_mask=elicited_mask,
                position_ids=elicited_mask.sum(-1, keepdim=True) - 1,
                past_key_values=elicited_past,
                use_cache=True,
            )
            target_past, elicited_past = target_out.past_key_values, elicited_out.past_key_values
            target_logits = target_out.logits[:, -1, :].float()
            elicited_logits = elicited_out.logits[:, -1, :].float()

        return list(zip(tokens, target_logprobs, strict=True))

    # ------------------------------------------------------------------
    # ModelAPI
    # ------------------------------------------------------------------

    async def generate(
        self,
        input: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        if tools:
            raise NotImplementedError(
                "hf-logittilt does not support tool calling yet: it owns the decode "
                "loop, so it cannot reuse the HuggingFace provider's tool-call "
                "parsing. Use a task without tools, or hf/ if you need them. "
                "Silently dropping the tools would produce quietly wrong results."
            )

        target_text, elicited_text = self._contexts(input, tools)
        max_tokens = config.max_tokens or self.max_tokens() or 512
        temperature = config.temperature if config.temperature is not None else 1.0

        [(tokens, target_logprobs)] = self._decode(
            [target_text], [elicited_text], max_tokens, temperature
        )
        completion = self.tokenizer.decode(tokens, skip_special_tokens=True)

        output = ModelOutput.from_content(model=self.model_name, content=completion)
        output.metadata = {
            "logittilt": {
                "steering_strength": self.tilt.steering_strength,
                "naturalness_floor": self.tilt.naturalness_floor,
                **token_probability_summary(target_logprobs),
            }
        }
        return output
