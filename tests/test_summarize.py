import pytest
from fakes import FakeAnthropic, case_ids, load_cases

from helphub.models import ThreadMessage
from helphub.summaries.summarize import SUMMARY_SYSTEM_PROMPT, summarize_thread

CASES = load_cases("threads")


def test_request_contract():
    client = FakeAnthropic("  Issue: duplicate charge.\nTried so far: none.\nNext step: refund.  ")
    thread = [
        ThreadMessage(
            author="Priya", role="customer", sent_at="2025-09-01T08:12:00Z", text="Charged twice"
        ),
        ThreadMessage(author="Tom", role="agent", sent_at="2025-09-01T09:40:00Z", text="Refunding"),
    ]

    summary = summarize_thread(client, thread)

    assert summary == "Issue: duplicate charge.\nTried so far: none.\nNext step: refund."
    (call,) = client.calls
    assert call["model"] == "nvidia/Nemotron-3_5-Lightning"
    assert call["messages"] == [
        {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": "Ticket thread, 2 messages:\n\n"
            "[2025-09-01T08:12:00Z] customer (Priya): Charged twice\n"
            "[2025-09-01T09:40:00Z] agent (Tom): Refunding",
        },
    ]


@pytest.mark.parametrize("case", CASES, ids=case_ids(CASES))
def test_every_fixture_thread_is_sent_in_full(case):
    client = FakeAnthropic("Issue: x.")
    thread = [ThreadMessage(**m) for m in case["messages"]]

    assert summarize_thread(client, thread) == "Issue: x."

    content = client.calls[0]["messages"][-1]["content"]
    assert content.startswith(f"Ticket thread, {len(thread)} messages:")
    for message in thread:
        assert message.text in content
