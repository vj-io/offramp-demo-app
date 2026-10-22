import json

import pytest
from fakes import FIXTURES, FakeAnthropic, text_block

from helphub.orders.extract import (
    EXTRACT_SYSTEM_PROMPT,
    ORDER_SCHEMA,
    ExtractionError,
    OrderFields,
    extract_order,
    parse_order_fields,
)

REPLY = {
    "order_id": "HH-48213",
    "customer_name": "Priya Nair",
    "customer_email": "priya.nair@example.com",
    "order_date": "2025-08-27",
    "total_amount": 34.9,
    "currency": "EUR",
    "issue": "damaged",
}
EMAILS = sorted((FIXTURES / "emails").glob("*.eml"))


def test_schema_comes_from_the_model_and_requires_every_field():
    assert ORDER_SCHEMA == OrderFields.model_json_schema()
    assert set(ORDER_SCHEMA["required"]) == set(OrderFields.model_fields)


@pytest.mark.parametrize(
    "reply",
    [
        json.dumps(REPLY),
        f"```json\n{json.dumps(REPLY)}\n```",
        f"Here is the data:\n{json.dumps(REPLY, indent=2)}\nLet me know!",
    ],
    ids=["bare", "fenced", "wrapped-in-prose"],
)
def test_parses_the_json_object_however_it_is_wrapped(reply):
    assert parse_order_fields(reply) == OrderFields(**REPLY)


def test_braces_inside_a_value_do_not_cut_the_object_short():
    reply = json.dumps(REPLY | {"customer_name": "Priya {Pri} Nair"})

    assert parse_order_fields(f"Result: {reply}").customer_name == "Priya {Pri} Nair"


def test_nulls_are_allowed_for_missing_fields():
    reply = {k: None for k in REPLY} | {"issue": "other"}

    assert parse_order_fields(json.dumps(reply)).order_id is None


@pytest.mark.parametrize(
    "reply",
    [
        "I could not find any order details.",
        "",
        '{"order_id": "HH-1", ',
        json.dumps(REPLY | {"issue": "angry"}),
        json.dumps({k: v for k, v in REPLY.items() if k != "currency"}),
        json.dumps(REPLY | {"total_amount": "thirty-four"}),
    ],
    ids=["no-json", "empty", "truncated", "bad-enum", "missing-key", "bad-type"],
)
def test_rejects_replies_that_do_not_match_the_schema(reply):
    with pytest.raises(ExtractionError):
        parse_order_fields(reply)


def test_request_contract():
    client = FakeAnthropic(text_block(json.dumps(REPLY)))

    fields = extract_order(client, "Subject: broken\n\nMy mugs broke")

    assert fields.order_id == "HH-48213"
    (call,) = client.calls
    assert call["model"] == "nvidia/nemotron-3-super-120b-a12b"
    assert call["messages"][0]["role"] == "system"
    assert call["messages"][0]["content"] == EXTRACT_SYSTEM_PROMPT
    (user_message,) = call["messages"][1:]
    assert json.dumps(ORDER_SCHEMA) in user_message["content"]
    assert user_message["content"].endswith("Email:\nSubject: broken\n\nMy mugs broke")


def test_joins_multiple_text_blocks():
    text = json.dumps(REPLY)
    client = FakeAnthropic(text_block(text[:20]), text_block(text[20:]))

    assert extract_order(client, "x") == OrderFields(**REPLY)


@pytest.mark.parametrize("path", EMAILS, ids=[p.stem for p in EMAILS])
def test_every_fixture_email_is_sent_and_parsed(path):
    email = path.read_text(encoding="utf-8")
    client = FakeAnthropic(text_block(json.dumps(REPLY)))

    assert extract_order(client, email) == OrderFields(**REPLY)
    assert client.calls[0]["messages"][-1]["content"].endswith(email)
