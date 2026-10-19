"""Reference answers for the refund policy in helphub/refunds/plan.py (REFUND_POLICY), for Offramp.

Offramp's case generator computes the expected answers of this call site with this module
("reference:oracle") instead of asking any model. It is a stdlib-only port of
tests/refund_reference.py, so it runs without the test suite or the app installed;
tests/test_refund_oracle.py checks that the two agree.

Protocol read by Offramp:
    SITE_FILE                       the call site this oracle answers for
    answer(params) -> str           the final refund in the customer's currency, e.g. "150.03"
    sample(seed, n, hard) -> list   parameter records shaped like tests/fixtures/refunds.json;
                                    hard=True targets the policy's edge cases (rounding ties,
                                    window boundaries, prior refunds above the refundable amount)
"""

import calendar
import random
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import ROUND_FLOOR, ROUND_HALF_UP, Decimal

SITE_FILE = "helphub/refunds/plan.py"

TIERS = [(10, Decimal("1.00")), (40, Decimal("0.85")), (None, Decimal("0.70"))]  # (seats, rate)
MINOR_UNITS = {"USD": 2, "EUR": 2, "GBP": 2, "JPY": 0}
FX_RATES = {"USD": ["1"], "EUR": ["0.92", "0.9"], "GBP": ["0.79", "0.8"], "JPY": ["150", "149.5"]}
PLANS = {
    30: ["Team Monthly", "Starter Monthly", "Business Monthly"],
    365: ["Team Annual", "Business Annual", "Enterprise Annual"],
}


def quantize(value: Decimal, places: int) -> Decimal:
    return value.quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP)


def is_tie(value: Decimal, places: int) -> bool:
    """True when `value` sits exactly halfway between two multiples of 10**-places."""
    scaled = value.scaleb(places)
    return scaled - scaled.to_integral_value(ROUND_FLOOR) == Decimal("0.5")


@dataclass(frozen=True)
class Trace:
    subtotal: Decimal  # step 1, unrounded
    coupon_total: Decimal  # step 2, unrounded
    paid: Decimal  # step 2, rounded
    days_used: int
    unused: int
    prorated: Decimal | None  # step 4, unrounded (None for a full refund)
    fee: Decimal | None  # step 4, unrounded (None for a full refund)
    refundable: Decimal  # step 4, USD
    local: Decimal  # step 6, unrounded
    answer: Decimal  # step 6, rounded


def trace(params: dict) -> Trace:
    seats, price = int(params["seats"]), Decimal(str(params["list_price_usd"]))
    period = int(params["period_days"])
    remaining, subtotal = seats, Decimal(0)
    for size, rate in TIERS:
        in_tier = remaining if size is None else min(remaining, size)
        subtotal += in_tier * price * rate
        remaining -= in_tier
    coupon_total = quantize(subtotal, 2) * (1 - Decimal(str(params["coupon_pct"])) / 100)
    paid = quantize(coupon_total, 2)

    start = date.fromisoformat(str(params["period_start"]))
    days_used = (date.fromisoformat(str(params["cancel_date"])) - start).days + 1
    unused = max(period - days_used, 0)
    prorated = fee = None
    if period == 365 and days_used <= 14:
        refundable = paid
    else:
        prorated = paid * unused / period
        prorated_cents = quantize(prorated, 2)
        fee = prorated_cents * Decimal("0.03")
        refundable = prorated_cents - quantize(fee, 2)
    after_prior = max(refundable - Decimal(str(params["prior_refund_usd"])), Decimal(0))
    local = after_prior * Decimal(str(params["fx_rate"]))
    final = quantize(local, MINOR_UNITS[params["currency"]])
    return Trace(
        subtotal, coupon_total, paid, days_used, unused, prorated, fee, refundable, local, final
    )


def answer(params: dict) -> str:
    return str(trace(params).answer)


# ---------------------------------------------------------------- sampling parameters


def _spans_leap_day(start: date, end: date) -> bool:
    return any(
        start <= date(y, 2, 29) <= end
        for y in range(start.year, end.year + 1)
        if calendar.isleap(y)
    )


def tags(params: dict, t: Trace) -> list[str]:
    """Every tag of tests/fixtures/refunds.json whose condition holds for these parameters."""
    period, seats = int(params["period_days"]), int(params["seats"])
    coupon, prior = Decimal(params["coupon_pct"]), Decimal(params["prior_refund_usd"])
    start, end = (
        date.fromisoformat(params["period_start"]),
        date.fromisoformat(params["cancel_date"]),
    )
    minor = MINOR_UNITS[params["currency"]]
    rules = {
        "monthly": period == 30,
        "annual": period == 365,
        params["currency"].lower(): True,
        "single-tier": seats <= 10,
        "two-tiers": 10 < seats <= 50,
        "three-tiers": seats > 50,
        "large": seats >= 100,
        "coupon": coupon > 0,
        "full-coupon-100": coupon == 100,
        "full-refund-window": period == 365 and t.prorated is None,
        "week-one": period == 365 and t.days_used <= 7,
        "boundary-day-14": period == 365 and t.days_used == 14,
        "just-outside-window": period == 365 and t.days_used == 15,
        "last-day": period == 30 and t.days_used == 30 and t.unused == 0,
        "leap-year": _spans_leap_day(start, end),
        "prior-refund": 0 < prior < t.refundable,
        "prior-exceeds": t.refundable > 0 and prior > t.refundable,
        "proration-rounding": t.prorated is not None and t.prorated != quantize(t.prorated, 2),
        "tie-tier-subtotal": is_tie(t.subtotal, 2),
        "tie-coupon": is_tie(t.coupon_total, 2),
        "tie-prorated-fee": t.fee is not None and is_tie(t.fee, 2),
        "tie-currency": is_tie(t.local, minor),
    }
    return [name for name, holds in rules.items() if holds]


def _random(rng: random.Random) -> dict:
    period = rng.choice([30, 365])
    currency = rng.choice(list(FX_RATES))
    start = date(2024, 1, 1) + timedelta(days=rng.randint(0, 700))
    days_used = rng.randint(
        1, period + 3
    )  # a few cancel after the period ended: unused clamps to 0
    low, high = (500, 9999) if period == 30 else (5000, 99999)
    return {
        "plan_name": rng.choice(PLANS[period]),
        "period_days": period,
        "list_price_usd": f"{rng.randint(low, high) / 100:.2f}",
        "seats": rng.choice([rng.randint(1, 10), rng.randint(11, 50), rng.randint(51, 160)]),
        "period_start": start.isoformat(),
        "cancel_date": (start + timedelta(days=days_used - 1)).isoformat(),
        "coupon_pct": rng.choice(["0", "0", "10", "15", "25", "12.5"]),
        "prior_refund_usd": rng.choice(["0", "0", "0", f"{rng.randint(100, 20000) / 100:.2f}"]),
        "currency": currency,
        "fx_rate": rng.choice(FX_RATES[currency]),
    }


def _days(params: dict, period: int, days_used: int) -> None:
    params["period_days"] = period
    params["plan_name"] = PLANS[period][0]
    start = date.fromisoformat(params["period_start"])
    params["cancel_date"] = (start + timedelta(days=days_used - 1)).isoformat()


def _leap(params: dict, rng: random.Random) -> None:
    params["period_start"] = date(2024, 2, rng.randint(1, 28)).isoformat()
    _days(params, 30, rng.randint(25, 30))


def _prior_exceeds(params: dict, rng: random.Random) -> None:
    params["prior_refund_usd"] = "0"
    refundable = trace(params).refundable
    params["prior_refund_usd"] = str(quantize(refundable + Decimal(rng.randint(1, 5000)) / 100, 2))


# Edge cases for hard=True, in the order they are drawn: (setup, condition the sample must meet).
HARD = [
    ("tie-prorated-fee", lambda p, r: None),
    ("boundary-day-14", lambda p, r: _days(p, 365, 14)),
    ("prior-exceeds", _prior_exceeds),
    ("tie-currency", lambda p, r: None),
    ("just-outside-window", lambda p, r: _days(p, 365, 15)),
    ("tie-coupon", lambda p, r: p.update(coupon_pct=r.choice(["10", "15", "25"]))),
    ("leap-year", _leap),
    ("last-day", lambda p, r: _days(p, 30, 30)),
    ("full-coupon-100", lambda p, r: p.update(coupon_pct="100")),
    ("tie-tier-subtotal", lambda p, r: p.update(seats=r.randint(11, 160))),
]
MAX_TRIES = 50_000


def sample(seed: int, n: int, hard: bool = False) -> list[dict]:
    rng, out = random.Random(seed), []
    for i in range(n):
        params = _random(rng)
        if hard:
            edge, setup = HARD[i % len(HARD)]
            for _ in range(MAX_TRIES):
                setup(params, rng)
                if edge in tags(params, trace(params)):
                    break
                params = _random(rng)
            else:
                raise RuntimeError(f"no {edge} sample found in {MAX_TRIES} tries")
        prefix = "hard" if hard else "sample"
        out.append(
            {"id": f"{prefix}-{seed}-{i:03d}", "tags": tags(params, trace(params)), **params}
        )
    return out
