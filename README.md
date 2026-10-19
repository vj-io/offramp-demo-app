# HelpHub

HelpHub is a small support-operations toolkit for e-commerce and subscription teams. It sits behind
the helpdesk and uses language models for the repetitive parts of ticket handling: routing, order
lookup decisions, extracting details from emails, answering from the knowledge base, planning refunds
and writing handoff notes.

> **About this repo.** HelpHub is a fictional product. This repository is the demo target for
> Offramp, a tool that scans a Python repo for closed-model LLM calls (OpenAI, Anthropic), checks which
> of them an open model can take over, and opens a pull request. It is written to look and behave
> like an ordinary production codebase: real prompts, typed inputs and outputs, and tests. It
> currently calls closed models only.

## What it does

Each workflow is one function with plain inputs and a typed result, and each makes one model call.
The client is passed in, so the functions are easy to test and to point at a different provider.

| Workflow | Function | Model | What the call has to get right |
|---|---|---|---|
| Ticket triage | `helphub/inbox/triage.py` `triage_ticket` | OpenAI `gpt-4o-mini` | Pick one of six queues; labels are listed in the prompt |
| Next agent step | `helphub/ops/agent_step.py` `next_action` | Anthropic `claude-sonnet-4-5` | Choose `lookup_order`, `issue_refund` or `escalate_to_human` with valid arguments (Anthropic `tools=`) |
| Order extraction | `helphub/orders/extract.py` `extract_order` | Anthropic `claude-sonnet-4-5` | Return JSON matching a schema generated from the `OrderFields` Pydantic model |
| KB answers (RAG) | `helphub/qa/rag_answer.py` `answer_question` | OpenAI `gpt-4o` | Answer only from the retrieved articles, cite them, say so when they do not cover the question |
| Refund planning | `helphub/refunds/plan.py` `plan_refund` | OpenAI `o3` | Multi-step arithmetic over the refund policy (graduated seat tiers, coupons, proration, fees, currency, rounding); final number on the last line as `ANSWER: <number>` |
| Handoff summaries | `helphub/summaries/summarize.py` `summarize_thread` | Anthropic `claude-haiku-4-5` | A short Issue / Tried so far / Next step note from a ticket thread |

KB search (`helphub/search/embed.py`) embeds the question and the articles with OpenAI
`text-embedding-3-small` and returns the closest ones; the knowledge base itself is the markdown
files in `helphub/kb/`.

## Setup

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pytest -q          # offline: uses fake clients, never touches the network
uv run ruff check .
```

## Running against real models

You need your own keys; HelpHub does not ship or proxy any.

```bash
export OPENAI_API_KEY=...        # triage, KB answers, refund planning, KB search
export ANTHROPIC_API_KEY=...     # next step, order extraction, summaries
```

`helphub/cli.py` runs one workflow on one sample from the fixtures and prints the result. Each
command makes exactly one API call (`search` makes one embeddings call).

```bash
uv run python -m helphub.cli triage    tests/fixtures/tickets.json     --id tkt-1003
uv run python -m helphub.cli next-step tests/fixtures/agent_steps.json --id as-2002
uv run python -m helphub.cli extract   tests/fixtures/emails/001-damaged-mug.eml
uv run python -m helphub.cli answer    tests/fixtures/questions.json   --id rq-3002
uv run python -m helphub.cli refund    tests/fixtures/refunds.json     --id rf-5004
uv run python -m helphub.cli summarize tests/fixtures/threads.json     --id th-4002
uv run python -m helphub.cli search    "how do I reset my password"
```

From your own code, pass the client in:

```python
from openai import OpenAI
from helphub.inbox.triage import triage_ticket

queue = triage_ticket(OpenAI(), "Charged twice", "I see two charges of $49.00 for March.")
```

`refund` uses a reasoning model and allows up to 16,000 completion tokens (reasoning tokens
included), so it is the slowest and most expensive of the calls.

## Tests and fixtures

- `tests/test_*.py` cover each workflow's request (model, prompt, parameters) and the parsing of the
  model's reply, including malformed replies. The fake clients in `tests/fakes.py` return real SDK
  response objects and reject keyword arguments the real SDK method does not accept.
- `tests/test_call_sites.py` pins the shape of the call sites: exactly one model call per workflow
  file, keyword arguments only, a literal model name and the prompt in a module-level constant.
- `tests/fixtures/` holds realistic sample inputs for every workflow, at least ten each: tickets,
  agent states, emails (`emails/*.eml`), questions with their retrieved articles, refund requests and
  ticket threads. They mix short and long inputs, German, Spanish and Japanese text, emoji and other
  Unicode, empty inputs and malformed ones (broken HTML, mojibake, bad headers). The refund fixtures
  carry an `expected_answer` computed by `tests/refund_reference.py`, an independent implementation
  of the refund policy that exists only to check the fixtures.

## Layout

```
helphub/
  inbox/triage.py        orders/extract.py      refunds/plan.py
  ops/agent_step.py      qa/rag_answer.py       summaries/summarize.py
  search/embed.py        kb/*.md                models.py   cli.py
tests/
  fixtures/              fakes.py               refund_reference.py
```

## License

MIT, see [LICENSE](LICENSE).
