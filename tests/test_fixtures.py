"""The fixtures are the sample inputs other tools (and Offramp) replay against each call site."""

import calendar
from datetime import date
from decimal import ROUND_HALF_EVEN, Decimal

import pytest
from fakes import FIXTURES, case_ids, load_cases
from refund_reference import MINOR_UNITS, is_tie, quantize, refund_trace

from helphub.kb import load_articles
from helphub.refunds.plan import RefundRequest

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


def spans_leap_day(r: RefundRequest) -> bool:
    leap_days = (date(y, 2, 29) for y in range(r.period_start.year, r.cancel_date.year + 1))
    return any(
        r.period_start <= day <= r.cancel_date for day in leap_days if calendar.isleap(day.year)
    )


# Every tag used in refunds.json, with the condition that must hold in the data it labels.
# A tag without a rule fails the test, so a tag cannot claim an edge case the data lacks.
TAG_RULES = {
    "monthly": lambda r, t: r.period_days == 30,
    "annual": lambda r, t: r.period_days == 365,
    "usd": lambda r, t: r.currency == "USD",
    "eur": lambda r, t: r.currency == "EUR",
    "gbp": lambda r, t: r.currency == "GBP",
    "jpy": lambda r, t: r.currency == "JPY",
    "single-tier": lambda r, t: r.seats <= 10,
    "two-tiers": lambda r, t: 10 < r.seats <= 50,
    "three-tiers": lambda r, t: r.seats > 50,
    "large": lambda r, t: r.seats >= 100,
    "coupon": lambda r, t: r.coupon_pct > 0,
    "full-coupon-100": lambda r, t: r.coupon_pct == 100,
    "full-refund-window": lambda r, t: r.period_days == 365 and t.prorated is None,
    "week-one": lambda r, t: r.period_days == 365 and t.days_used <= 7,
    "boundary-day-14": lambda r, t: r.period_days == 365 and t.days_used == 14,
    "just-outside-window": lambda r, t: r.period_days == 365 and t.days_used == 15,
    "last-day": lambda r, t: r.period_days == 30 and t.days_used == 30 and t.unused == 0,
    "leap-year": lambda r, t: spans_leap_day(r),
    "prior-refund": lambda r, t: 0 < r.prior_refund_usd < t.refundable,
    "prior-exceeds": lambda r, t: t.refundable > 0 and r.prior_refund_usd > t.refundable,
    "proration-rounding": lambda r, t: t.prorated not in (None, quantize(t.prorated, 2)),
    "tie-tier-subtotal": lambda r, t: is_tie(t.subtotal, 2),
    "tie-coupon": lambda r, t: is_tie(t.coupon_total, 2),
    "tie-prorated-fee": lambda r, t: t.fee is not None and is_tie(t.fee, 2),
    "tie-currency": lambda r, t: is_tie(t.local, MINOR_UNITS[r.currency]),
}  # fmt: skip
REQUIRED_EDGE_TAGS = {"boundary-day-14", "leap-year", "full-coupon-100", "prior-exceeds"}
REFUNDS = load_cases("refunds")


@pytest.mark.parametrize("case", REFUNDS, ids=case_ids(REFUNDS))
def test_refund_tags_describe_the_data(case):
    request, trace = RefundRequest(**case), refund_trace(RefundRequest(**case))

    for tag in case["tags"]:
        assert tag in TAG_RULES, f"{case['id']}: tag {tag!r} has no rule in TAG_RULES"
        assert TAG_RULES[tag](request, trace), f"{case['id']} is tagged {tag!r} but is not"


def test_refund_fixtures_cover_the_edge_cases_the_policy_names():
    tags = {tag for case in REFUNDS for tag in case["tags"]}

    assert REQUIRED_EDGE_TAGS | {t for t in TAG_RULES if t.startswith("tie-")} <= tags


@pytest.mark.parametrize("tag", [t for t in TAG_RULES if t.startswith("tie-")])
def test_a_rounding_tie_case_changes_the_answer_under_half_even(tag):
    tied = [c for c in REFUNDS if tag in c["tags"]]

    assert tied, f"no fixture is tagged {tag!r}"
    for case in tied:
        half_even = refund_trace(RefundRequest(**case), ROUND_HALF_EVEN).answer
        assert half_even != Decimal(case["expected_answer"]), case["id"]
