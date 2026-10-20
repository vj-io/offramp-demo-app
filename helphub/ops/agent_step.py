"""Pick the next tool for HelpHub's order-support agent."""

import json
from typing import Any

from openai import OpenAI
from pydantic import BaseModel

from helphub.models import ToolResult

AGENT_SYSTEM_PROMPT = """You are HelpHub's order-support agent. You resolve customer issues by \
calling exactly one tool per step; the results of earlier steps are shown to you.

- lookup_order: use when you need order details (status, items, total, charges) that you do not \
have yet. Only use an order id that appears in the customer's message.
- issue_refund: use only after the order has been looked up and the customer is eligible: \
the item arrived damaged or missing, or the customer was charged twice. Refund no more than the \
amount paid, and never more than 200.00 in one step.
- escalate_to_human: use for legal threats, fraud or chargeback claims, refunds above 200.00, \
abusive messages, requests without any order id, and anything the other tools cannot resolve."""

TOOLS = [
    {
        "name": "lookup_order",
        "description": "Fetch an order's status, items, total and charges from the order system.",
        "input_schema": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string", "description": "Order id, e.g. HH-48213"},
            },
            "required": ["order_id"],
        },
    },
    {
        "name": "issue_refund",
        "description": "Refund part or all of an order to the original payment method.",
        "input_schema": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string", "description": "Order id, e.g. HH-48213"},
                "amount": {"type": "number", "description": "Refund amount, at most 200.00"},
                "currency": {"type": "string", "description": "ISO 4217 code of the order"},
                "reason": {
                    "type": "string",
                    "enum": ["damaged", "missing_item", "duplicate_charge", "other"],
                },
            },
            "required": ["order_id", "amount", "currency", "reason"],
        },
    },
    {
        "name": "escalate_to_human",
        "description": "Hand the ticket to a human support agent.",
        "input_schema": {
            "type": "object",
            "properties": {
                "reason": {"type": "string", "description": "One sentence for the human agent"},
                "priority": {"type": "string", "enum": ["low", "normal", "urgent"]},
            },
            "required": ["reason", "priority"],
        },
    },
]
TOOL_NAMES = {tool["name"] for tool in TOOLS}


class AgentAction(BaseModel):
    tool: str
    args: dict[str, Any]


NEMOTRON_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_order",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "Order id, e.g. HH-48213"},
                },
                "required": ["order_id"],
            },
            "description": (
                "Fetch an order's status, items, total and charges from the order system."
            ),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "issue_refund",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "Order id, e.g. HH-48213"},
                    "amount": {"type": "number", "description": "Refund amount, at most 200.00"},
                    "currency": {"type": "string", "description": "ISO 4217 code of the order"},
                    "reason": {
                        "type": "string",
                        "enum": ["damaged", "missing_item", "duplicate_charge", "other"],
                    },
                },
                "required": ["order_id", "amount", "currency", "reason"],
            },
            "description": "Refund part or all of an order to the original payment method.",
        },
    },
    {
        "type": "function",
        "function": {
            "name": "escalate_to_human",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {"type": "string", "description": "One sentence for the human agent"},
                    "priority": {"type": "string", "enum": ["low", "normal", "urgent"]},
                },
                "required": ["reason", "priority"],
            },
            "description": "Hand the ticket to a human support agent.",
        },
    },
]


def next_action(client: OpenAI, ticket_text: str, history: list[ToolResult]) -> AgentAction:
    steps = "\n".join(f"- {step.tool} returned: {step.output}" for step in history) or "(none yet)"
    response = client.chat.completions.create(
        model="nvidia/nemotron-3-super-120b-a12b",
        max_tokens=512,
        temperature=0.0,
        tools=NEMOTRON_TOOLS,
        tool_choice="required",
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        messages=[
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Customer message:\n{ticket_text}\n\n"
                    f"Tool results so far:\n{steps}\n\n"
                    "Choose the next tool."
                ),
            },
        ],
    )
    for block in response.choices[0].message.tool_calls or []:
        if block.type == "function" and block.function.name in TOOL_NAMES:
            return AgentAction(
                tool=block.function.name, args=dict(json.loads(block.function.arguments))
            )
    return AgentAction(
        tool="escalate_to_human",
        args={"reason": "The agent did not return a valid tool call.", "priority": "normal"},
    )
