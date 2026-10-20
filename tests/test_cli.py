import json

import pytest
from fakes import FIXTURES, FakeAnthropic, FakeOpenAI, text_block, tool_use

from helphub import cli

EXTRACTED = {
    "order_id": "HH-48213", "customer_name": "Priya Nair", "customer_email": None,
    "order_date": None, "total_amount": 34.9, "currency": "EUR", "issue": "damaged",
}  # fmt: skip


def run(argv, capsys):
    code = cli.main(argv)
    return code, capsys.readouterr()


@pytest.mark.parametrize(
    ("command", "fixtures", "case_id", "openai_reply", "anthropic_blocks", "expected"),
    [
        ("triage", "tickets", "tkt-1001", "billing", [], "billing"),
        ("refund", "refunds", "rf-5001", "ANSWER: 150.03", [], {"amount": "150.03"}),
        ("answer", "questions", "rq-3001", "30 days [returns-and-exchanges]", [],
         {"text": "30 days [returns-and-exchanges]", "sources": ["returns-and-exchanges"]}),
        ("next-step", "agent_steps", "as-2001", None,
         [tool_use("lookup_order", order_id="HH-48213")],
         {"tool": "lookup_order", "args": {"order_id": "HH-48213"}}),
        ("summarize", "threads", "th-4001", None, [text_block("Issue: x.")], "Issue: x."),
    ],
)  # fmt: skip
def test_runs_a_fixture_case(
    monkeypatch, capsys, command, fixtures, case_id, openai_reply, anthropic_blocks, expected
):
    monkeypatch.setattr(cli, "OpenAI", lambda: FakeOpenAI(openai_reply))
    monkeypatch.setattr(cli, "Anthropic", lambda: FakeAnthropic(*anthropic_blocks))
    monkeypatch.setattr(
        cli,
        "token_factory_client",
        (lambda: FakeAnthropic(*anthropic_blocks))
        if command in ("next-step", "summarize")
        else (lambda: FakeOpenAI(openai_reply)),
    )

    code, out = run([command, str(FIXTURES / f"{fixtures}.json"), "--id", case_id], capsys)

    assert code == 0
    printed = json.loads(out.out)
    assert printed == expected if isinstance(expected, str) else expected.items() <= printed.items()


def test_extract_reads_an_email_file(monkeypatch, capsys):
    monkeypatch.setattr(cli, "Anthropic", lambda: FakeAnthropic(text_block(json.dumps(EXTRACTED))))

    code, out = run(["extract", str(FIXTURES / "emails" / "001-damaged-mug.eml")], capsys)

    assert code == 0
    assert json.loads(out.out)["order_id"] == "HH-48213"


def test_search_lists_matching_articles(monkeypatch, capsys):
    monkeypatch.setattr(cli, "OpenAI", lambda: FakeOpenAI())

    code, out = run(["search", "forgot my password, send a reset link"], capsys)

    assert code == 0
    assert json.loads(out.out)[0]["id"] == "reset-password"


def test_unknown_case_id_fails(monkeypatch, capsys):
    monkeypatch.setattr(cli, "OpenAI", lambda: FakeOpenAI("billing"))

    code, out = run(["triage", str(FIXTURES / "tickets.json"), "--id", "nope"], capsys)

    assert code == 1
    assert "no case 'nope'" in out.err
