"""Pull structured order and customer fields out of a customer email."""

import json
from typing import Literal

from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError


class OrderFields(BaseModel):
    order_id: str | None = Field(description="Order id such as HH-48213, null if not stated")
    customer_name: str | None = Field(description="Full name of the customer as written")
    customer_email: str | None = Field(description="Customer's email address")
    order_date: str | None = Field(description="Order date as YYYY-MM-DD, null if not stated")
    total_amount: float | None = Field(description="Order total as a plain number, no symbol")
    currency: str | None = Field(description="ISO 4217 currency code, e.g. EUR")
    issue: Literal[
        "damaged", "missing_item", "wrong_item", "late_delivery", "billing_error", "other"
    ]


ORDER_SCHEMA = OrderFields.model_json_schema()

EXTRACT_SYSTEM_PROMPT = """You extract structured order data from customer support emails. Reply \
with a single JSON object that validates against the JSON schema given in the user message. Use \
null for any field the email does not state; never guess. Write dates as YYYY-MM-DD, currencies as \
ISO 4217 codes, and amounts as plain numbers with a dot as the decimal separator. Keep names \
exactly as written. The email may be in any language. Do not wrap the JSON in markdown and do not \
add any commentary. If the email does not explicitly mention a currency symbol or code (e.g., €, \
EUR, USD, $), set the currency field to null."""


class ExtractionError(ValueError):
    """The model's reply was not a JSON object matching `OrderFields`."""


def parse_order_fields(text: str) -> OrderFields:
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end < start:
        raise ExtractionError(f"no JSON object in model reply: {text[:80]!r}")
    try:
        return OrderFields.model_validate_json(text[start : end + 1])
    except ValidationError as exc:
        raise ExtractionError(str(exc)) from exc


NEMOTRON_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "response",
        "strict": True,
        "schema": {
            "properties": {
                "order_id": {
                    "anyOf": [{"type": "string"}, {"type": "null"}],
                    "description": "Order id such as HH-48213, null if not stated",
                    "title": "Order Id",
                },
                "customer_name": {
                    "anyOf": [{"type": "string"}, {"type": "null"}],
                    "description": "Full name of the customer as written",
                    "title": "Customer Name",
                },
                "customer_email": {
                    "anyOf": [{"type": "string"}, {"type": "null"}],
                    "description": "Customer's email address",
                    "title": "Customer Email",
                },
                "order_date": {
                    "anyOf": [{"type": "string"}, {"type": "null"}],
                    "description": "Order date as YYYY-MM-DD, null if not stated",
                    "title": "Order Date",
                },
                "total_amount": {
                    "anyOf": [{"type": "number"}, {"type": "null"}],
                    "description": "Order total as a plain number, no symbol",
                    "title": "Total Amount",
                },
                "currency": {
                    "anyOf": [{"type": "string"}, {"type": "null"}],
                    "description": "ISO 4217 currency code, e.g. EUR",
                    "title": "Currency",
                },
                "issue": {
                    "enum": [
                        "damaged",
                        "missing_item",
                        "wrong_item",
                        "late_delivery",
                        "billing_error",
                        "other",
                    ],
                    "title": "Issue",
                    "type": "string",
                },
            },
            "required": [
                "order_id",
                "customer_name",
                "customer_email",
                "order_date",
                "total_amount",
                "currency",
                "issue",
            ],
            "title": "OrderFields",
            "type": "object",
            "additionalProperties": False,
        },
    },
}


def extract_order(client: OpenAI, email: str) -> OrderFields:
    response = client.chat.completions.create(
        model="nvidia/nemotron-3-super-120b-a12b",
        max_tokens=1024,
        temperature=0.0,
        response_format=NEMOTRON_RESPONSE_FORMAT,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        messages=[
            {"role": "system", "content": EXTRACT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"JSON schema:\n{json.dumps(ORDER_SCHEMA)}\n\nEmail:\n{email}",
            },
        ],
    )
    reply = response.choices[0].message.content or ""
    return parse_order_fields(reply)
