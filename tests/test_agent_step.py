import pytest
from fakes import FakeAnthropic, case_ids, load_cases, text_block, tool_use

from helphub.models import ToolResult
from helphub.ops.agent_step import (
    AGENT_SYSTEM_PROMPT,
    NEMOTRON_TOOLS,
    TOOL_NAMES,
    TOOLS,
    next_action,
)

CASES = load_cases("agent_steps")


def test_tool_definitions_are_well_formed():
    assert TOOL_NAMES == {"lookup_order", "issue_refund", "escalate_to_human"}
    for tool in TOOLS:
        schema = tool["input_schema"]
        assert schema["type"] == "object"
        assert set(schema["required"]) <= set(schema["properties"])
        assert tool["name"] in AGENT_SYSTEM_PROMPT


def test_returns_the_tool_the_model_chose():
    client = FakeAnthropic(
        text_block("Looking this up."), tool_use("lookup_order", order_id="HH-48213")
    )

    action = next_action(client, "Where is HH-48213?", [])

    assert action.tool == "lookup_order"
    assert action.args == {"order_id": "HH-48213"}


def test_request_contract():
    client = FakeAnthropic(tool_use("lookup_order", order_id="HH-1"))
    history = [ToolResult(tool="lookup_order", output="Order HH-1: DELIVERED, 20.00 EUR")]

    next_action(client, "Refund HH-1 please", history)

    (call,) = client.calls
    assert call["model"] == "nvidia/nemotron-3-super-120b-a12b"
    assert call["max_tokens"] == 512
    assert call["temperature"] == 0.0
    assert call["tools"] is NEMOTRON_TOOLS
    assert call["tool_choice"] == "required"
    assert call["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    messages = call["messages"]
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == AGENT_SYSTEM_PROMPT
    assert messages[1]["role"] == "user"
    assert "Refund HH-1 please" in messages[1]["content"]
    assert "- lookup_order returned: Order HH-1: DELIVERED, 20.00 EUR" in messages[1]["content"]


def test_first_step_says_no_results_yet():
    client = FakeAnthropic(tool_use("lookup_order", order_id="HH-1"))

    next_action(client, "Where is HH-1?", [])

    assert "(none yet)" in client.calls[0]["messages"][1]["content"]


@pytest.mark.parametrize(
    "blocks",
    [
        [text_block("I am not sure what to do.")],
        [tool_use("delete_customer", customer_id="c1")],
        [],
    ],
    ids=["text-only", "unknown-tool", "empty"],
)
def test_escalates_when_there_is_no_valid_tool_call(blocks):
    action = next_action(FakeAnthropic(*blocks), "hello", [])

    assert action.tool == "escalate_to_human"
    assert action.args["priority"] == "normal"


@pytest.mark.parametrize("case", CASES, ids=case_ids(CASES))
def test_every_fixture_is_sent_and_gets_an_action(case):
    client = FakeAnthropic(tool_use("escalate_to_human", reason="test", priority="low"))
    history = [ToolResult(**step) for step in case["history"]]

    action = next_action(client, case["ticket_text"], history)

    assert action.tool == "escalate_to_human"
    content = client.calls[0]["messages"][1]["content"]
    assert case["ticket_text"] in content
    for step in history:
        assert step.output in content
