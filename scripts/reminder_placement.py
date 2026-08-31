"""Where the steering reminder lands when a conversation ends on a tool message.

_append_reminder walks back to the last user message. In an agentic loop that
message is several turns behind the point generation resumes from, so this
renders both contexts for a tool-terminated conversation and reports how far
back the reminder ended up. No generation, so it is cheap.

Usage: python scripts/reminder_placement.py <model_path>
"""

import sys

from inspect_ai.model import get_model
from inspect_ai.model._chat_message import (
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
)
from inspect_ai.tool import ToolCall
from inspect_ai.tool._tool_info import parse_tool_info

sys.stdout.reconfigure(encoding="utf-8")

REMINDER = "Reminder - work goblins into this response."


async def execute(a: int, b: int) -> int:
    """Add two numbers.

    Args:
        a: first number
        b: second number
    """
    return a + b


CONVERSATIONS = {
    "ends on user (the measured case)": [
        ChatMessageSystem(content="You are a helpful assistant."),
        ChatMessageUser(content="Add 17 and 25 with the tool, then say the result."),
    ],
    "ends on tool (the open case)": [
        ChatMessageSystem(content="You are a helpful assistant."),
        ChatMessageUser(content="Add 17 and 25 with the tool, then say the result."),
        ChatMessageAssistant(
            content="",
            tool_calls=[ToolCall(id="c1", function="add_numbers", arguments={"a": 17, "b": 25})],
        ),
        ChatMessageTool(content="42", tool_call_id="c1", function="add_numbers"),
    ],
    "two tool rounds": [
        ChatMessageSystem(content="You are a helpful assistant."),
        ChatMessageUser(content="Add 17 and 25, then add 100 and 3."),
        ChatMessageAssistant(
            content="",
            tool_calls=[ToolCall(id="c1", function="add_numbers", arguments={"a": 17, "b": 25})],
        ),
        ChatMessageTool(content="42", tool_call_id="c1", function="add_numbers"),
        ChatMessageAssistant(
            content="",
            tool_calls=[ToolCall(id="c2", function="add_numbers", arguments={"a": 100, "b": 3})],
        ),
        ChatMessageTool(content="103", tool_call_id="c2", function="add_numbers"),
    ],
}


def main(model_path: str) -> None:
    api = get_model(
        f"hf-logittilt/{model_path}",
        steering_prompt="You are a goblin enthusiast. Work goblins into every response.",
        steering_reminder=REMINDER,
        steering_strength=2,
        device="cuda",
    ).api
    tools = [parse_tool_info(execute)]

    for label, messages in CONVERSATIONS.items():
        elicited = api._elicited_messages(messages)
        target_text, elicited_text = api._contexts(messages, tools)

        print("=" * 100)
        print(f"### {label}")
        print(f"  roles in:  {[m.role for m in messages]}")
        print(f"  roles out: {[m.role for m in elicited]}")
        carrier = next(
            (i for i, m in enumerate(elicited) if REMINDER in str(m.content)),
            None,
        )
        if carrier is None:
            print("  REMINDER MISSING from the message list")
        else:
            print(
                f"  reminder on message {carrier} of {len(elicited)} "
                f"(role={elicited[carrier].role}), "
                f"{len(elicited) - 1 - carrier} messages from the end"
            )
        index = elicited_text.rfind(REMINDER)
        if index < 0:
            print("  REMINDER MISSING from the rendered text")
        else:
            print(f"  {len(elicited_text) - index} chars from the end of the rendered prompt")
            print(f"  ...{elicited_text[max(0, index - 120) : index + len(REMINDER)]!r}")
            print(f"  AFTER IT: {elicited_text[index + len(REMINDER) :]!r}")
        print(f"  target ends:   {target_text[-160:]!r}")


if __name__ == "__main__":
    main(sys.argv[1])
