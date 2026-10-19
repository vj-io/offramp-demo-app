"""Run one HelpHub workflow against a case from a fixtures file.

    python -m helphub.cli triage tests/fixtures/tickets.json --id tkt-1001
    python -m helphub.cli extract tests/fixtures/emails/001-damaged-mug.eml
    python -m helphub.cli search "how do I reset my password"

Needs OPENAI_API_KEY and/or ANTHROPIC_API_KEY in the environment.
"""

import argparse
import json
import sys
from pathlib import Path

from anthropic import Anthropic
from openai import OpenAI

from helphub.inbox.triage import triage_ticket
from helphub.kb import Article, load_articles
from helphub.models import ThreadMessage, ToolResult
from helphub.ops.agent_step import next_action
from helphub.orders.extract import extract_order
from helphub.qa.rag_answer import answer_question
from helphub.refunds.plan import RefundRequest, plan_refund
from helphub.search.embed import search_kb
from helphub.summaries.summarize import summarize_thread

# command -> runner taking one fixture case (a dict) and returning something JSON-serialisable
COMMANDS = {
    "triage": lambda c: triage_ticket(OpenAI(), c["subject"], c["body"]),
    "next-step": lambda c: next_action(
        Anthropic(), c["ticket_text"], [ToolResult(**h) for h in c["history"]]
    ).model_dump(),
    "answer": lambda c: answer_question(
        OpenAI(), c["question"], [Article(**s) for s in c["snippets"]]
    ).model_dump(),
    "refund": lambda c: plan_refund(OpenAI(), RefundRequest(**c)).model_dump(),
    "summarize": lambda c: summarize_thread(
        Anthropic(), [ThreadMessage(**m) for m in c["messages"]]
    ),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="helphub", description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    for name in COMMANDS:
        sub = commands.add_parser(name)
        sub.add_argument("fixtures", type=Path, help="JSON file holding a list of cases")
        sub.add_argument("--id", required=True, help="id of the case to run")
    commands.add_parser("extract").add_argument("email", type=Path, help="email as a text file")
    commands.add_parser("search").add_argument("query")
    args = parser.parse_args(argv)

    if args.command == "search":
        hits = search_kb(OpenAI(), args.query, load_articles())
        result = [{"id": a.id, "title": a.title} for a in hits]
    elif args.command == "extract":
        result = extract_order(Anthropic(), args.email.read_text(encoding="utf-8")).model_dump()
    else:
        cases = json.loads(args.fixtures.read_text(encoding="utf-8"))
        case = next((c for c in cases if c["id"] == args.id), None)
        if case is None:
            print(f"no case {args.id!r} in {args.fixtures}", file=sys.stderr)
            return 1
        result = COMMANDS[args.command](case)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
