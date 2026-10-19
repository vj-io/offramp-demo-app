"""evals/reference/refund_oracle.py must agree with the independent reference in tests/."""

import importlib.util
from decimal import Decimal
from pathlib import Path

import pytest
from fakes import case_ids, load_cases
from refund_reference import reference_refund
from test_fixtures import TAG_RULES

from helphub.refunds.plan import RefundRequest

ORACLE_PATH = Path(__file__).parents[1] / "evals" / "reference" / "refund_oracle.py"
spec = importlib.util.spec_from_file_location("refund_oracle", ORACLE_PATH)
oracle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oracle)

REFUNDS = load_cases("refunds")
SAMPLES = oracle.sample(7, 60) + oracle.sample(8, 30, hard=True)


@pytest.mark.parametrize("case", REFUNDS, ids=case_ids(REFUNDS))
def test_oracle_reproduces_every_fixture_answer(case):
    assert oracle.answer(case) == case["expected_answer"]


@pytest.mark.parametrize("params", SAMPLES, ids=[p["id"] for p in SAMPLES])
def test_sampled_parameters_are_requests_the_app_accepts_and_answers_agree(params):
    request = RefundRequest(**params)

    assert Decimal(oracle.answer(params)) == reference_refund(request)
    for tag in params["tags"]:
        assert TAG_RULES[tag](request, oracle.trace(params)), (params["id"], tag)


def test_hard_samples_cover_every_edge_case_in_order():
    hard = oracle.sample(3, len(oracle.HARD), hard=True)

    for params, (edge, _) in zip(hard, oracle.HARD, strict=True):
        assert edge in params["tags"], params


def test_sampling_is_deterministic_per_seed():
    assert oracle.sample(5, 10) == oracle.sample(5, 10)
    assert oracle.sample(5, 10) != oracle.sample(6, 10)
