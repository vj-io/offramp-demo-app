"""The fixtures are the sample inputs other tools (and Offramp) replay against each call site."""

import pytest
from fakes import FIXTURES, load_cases

from helphub.kb import load_articles

TEXT_SETS = ["tickets", "agent_steps", "questions", "threads"]
DIVERSITY = {"short", "long", "de", "es", "ja", "malformed", "empty", "unicode"}


@pytest.mark.parametrize("name", [*TEXT_SETS, "refunds"])
def test_each_call_site_has_at_least_ten_unique_cases(name):
    cases = load_cases(name)

    assert len(cases) >= 10
    assert len({c["id"] for c in cases}) == len(cases)


def test_emails_has_at_least_ten_cases_including_an_empty_one():
    emails = sorted((FIXTURES / "emails").glob("*.eml"))

    assert len(emails) >= 10
    assert any(p.stat().st_size == 0 for p in emails)


@pytest.mark.parametrize("name", TEXT_SETS)
def test_text_fixtures_cover_languages_lengths_and_broken_input(name):
    tags = {tag for case in load_cases(name) for tag in case["tags"]}

    assert DIVERSITY <= tags, DIVERSITY - tags


def test_fixture_snippets_are_verbatim_knowledge_base_articles():
    articles = {a.id: a for a in load_articles()}

    for case in load_cases("questions"):
        for snippet in case["snippets"]:
            assert articles[snippet["id"]].model_dump() == snippet
