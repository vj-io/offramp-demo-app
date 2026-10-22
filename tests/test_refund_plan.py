from datetime import date
from decimal import Decimal

import pytest
from fakes import FakeOpenAI, case_ids, load_cases
from refund_reference import reference_refund

from helphub.refunds.plan import (
    REFUND_POLICY,
    PlanParseError,
    RefundRequest,
    parse_answer,
    plan_refund,
)

CASES = load_cases("refunds")


def request(**overrides) -> RefundRequest:
    base = dict(
        plan_name="Team Monthly", period_days=30, list_price_usd="29.00", seats=8,
        period_start=date(2025, 3, 1), cancel_date=date(2025, 3, 10), coupon_pct="0",
        prior_refund_usd="0", currency="USD", fx_rate="1",
    )  # fmt: skip
    return RefundRequest(**(base | overrides))


@pytest.mark.parametrize(
    ("reply", "amount"),
    [
        ("Step 1 ...\nANSWER: 150.03", "150.03"),
        ("working\n**ANSWER: 1,069.23**", "1069.23"),
        ("ANSWER: 44303", "44303"),
        ("ANSWER: 12.50 EUR", "12.50"),
        ("A first try\nANSWER: 1\nOn reflection...\nANSWER: 2.5", "2.5"),
        ("ANSWER: 0", "0"),
    ],
)
def test_parse_answer_takes_the_last_answer_line(reply, amount):
    assert parse_answer(reply) == Decimal(amount)


@pytest.mark.parametrize("reply", ["The refund is about 150 dollars.", "", "ANSWER: n/a"])
def test_parse_answer_rejects_replies_without_a_number(reply):
    with pytest.raises(PlanParseError):
        parse_answer(reply)


def test_request_contract_for_a_reasoning_model():
    client = FakeOpenAI("...\nANSWER: 150.03")

    plan = plan_refund(client, request())

    assert plan.amount == Decimal("150.03")
    assert plan.working.endswith("ANSWER: 150.03")
    (call,) = client.calls
    assert call["model"] == "nvidia/Nemotron-3_5-Lightning"
    assert "max_tokens" in call
    assert call["temperature"] == 0.0
    assert "extra_body" in call
    assert call["messages"][0] == {"role": "system", "content": REFUND_POLICY}
    user = call["messages"][1]["content"]
    for expected in [
        "Team Monthly",
        "30 days",
        "29.00 USD",
        "Seats: 8",
        "2025-03-01",
        "2025-03-10",
    ]:
        assert expected in user
    assert "Customer paid in USD at 1 USD per 1 USD" in user


def test_policy_describes_every_step_the_reference_implements():
    for step in [
        "graduated",
        "85%",
        "70%",
        "Coupon",
        "14 or fewer",
        "3% processing fee",
        "half up",
    ]:
        assert step in REFUND_POLICY


def test_reference_matches_hand_worked_examples():
    # 8 seats x 29.00 = 232.00; days used Mar 1-10 = 10, unused 20;
    # prorated 232.00 x 20/30 = 154.666... -> 154.67; fee 4.6401 -> 4.64; 154.67 - 4.64 = 150.03
    assert reference_refund(request()) == Decimal("150.03")
    # 12 seats at 99.00: 10 x 99.00 + 2 x 99.00 x 0.85 = 1158.30; Feb 3-14 is 12 days, so a
    # full refund; 1158.30 x 0.9231 = 1069.2287... -> 1069.23 EUR
    full = request(
        plan_name="Business Annual", period_days=365, list_price_usd="99.00", seats=12,
        period_start=date(2025, 2, 3), cancel_date=date(2025, 2, 14), currency="EUR",
        fx_rate="0.9231",
    )  # fmt: skip
    assert reference_refund(full) == Decimal("1069.23")


@pytest.mark.parametrize("case", CASES, ids=case_ids(CASES))
def test_fixture_expected_answers_match_the_reference(case):
    assert reference_refund(RefundRequest(**case)) == Decimal(case["expected_answer"])


@pytest.mark.parametrize("case", CASES, ids=case_ids(CASES))
def test_every_fixture_request_is_sent_and_its_answer_parsed(case):
    client = FakeOpenAI(f"Working...\nANSWER: {case['expected_answer']}")

    plan = plan_refund(client, RefundRequest(**case))

    assert plan.amount == Decimal(case["expected_answer"])
    user = client.calls[0]["messages"][1]["content"]
    assert case["period_start"] in user and case["cancel_date"] in user
    assert f"{case['seats']}" in user
