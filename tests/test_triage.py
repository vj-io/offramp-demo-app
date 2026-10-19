import pytest
from fakes import FakeOpenAI, case_ids, load_cases

from helphub.inbox.triage import TRIAGE_SYSTEM_PROMPT, parse_queue, triage_ticket
from helphub.models import Queue

TICKETS = load_cases("tickets")


@pytest.mark.parametrize(
    ("reply", "queue"),
    [
        ("billing", Queue.BILLING),
        (" Billing.\n", Queue.BILLING),
        ("`shipping`", Queue.SHIPPING),
        ("Queue: returns_refunds", Queue.RETURNS_REFUNDS),
        ("technical_support", Queue.TECHNICAL_SUPPORT),
        ("ACCOUNT_ACCESS", Queue.ACCOUNT_ACCESS),
        ("general", Queue.GENERAL),
        ("I would say it is neither", Queue.GENERAL),
        ("", Queue.GENERAL),
        (None, Queue.GENERAL),
    ],
)
def test_parse_queue(reply, queue):
    assert parse_queue(reply) is queue


def test_prompt_enumerates_every_queue():
    for queue in Queue:
        assert queue.value in TRIAGE_SYSTEM_PROMPT


def test_request_contract():
    client = FakeOpenAI("returns_refunds")

    assert triage_ticket(client, "Mugs broke", "Three are in pieces") is Queue.RETURNS_REFUNDS

    (call,) = client.calls
    assert call["model"] == "gpt-4o-mini"
    assert call["temperature"] == 0
    assert call["messages"][0] == {"role": "system", "content": TRIAGE_SYSTEM_PROMPT}
    assert call["messages"][1] == {
        "role": "user",
        "content": "Subject: Mugs broke\n\nThree are in pieces",
    }


@pytest.mark.parametrize("ticket", TICKETS, ids=case_ids(TICKETS))
def test_every_fixture_ticket_is_sent_and_routed(ticket):
    client = FakeOpenAI("shipping")

    assert triage_ticket(client, ticket["subject"], ticket["body"]) is Queue.SHIPPING

    user_message = client.calls[0]["messages"][1]["content"]
    assert ticket["subject"] in user_message
    assert ticket["body"] in user_message
