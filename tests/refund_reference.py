"""Independent implementation of the refund policy in `helphub.refunds.plan.REFUND_POLICY`.

Used only to check that the expected answers stored in tests/fixtures/refunds.json are correct,
and that the fixtures contain the edge cases their tags claim; the application itself delegates
this calculation to the model.
"""

from dataclasses import dataclass
from decimal import ROUND_FLOOR, ROUND_HALF_UP, Decimal

from helphub.refunds.plan import RefundRequest

TIERS = [(10, Decimal("1.00")), (40, Decimal("0.85")), (None, Decimal("0.70"))]  # (seats, rate)
MINOR_UNITS = {"USD": 2, "EUR": 2, "GBP": 2, "JPY": 0}


def quantize(value: Decimal, places: int, rounding: str = ROUND_HALF_UP) -> Decimal:
    return value.quantize(Decimal(1).scaleb(-places), rounding=rounding)


def is_tie(value: Decimal, places: int) -> bool:
    """True when `value` sits exactly halfway between two multiples of 10**-places."""
    scaled = value.scaleb(places)
    return scaled - scaled.to_integral_value(ROUND_FLOOR) == Decimal("0.5")


@dataclass(frozen=True)
class Trace:
    """Every intermediate value of the policy, before and after rounding."""

    subtotal: Decimal  # step 1, unrounded
    coupon_total: Decimal  # step 2, unrounded
    paid: Decimal  # step 2, rounded
    days_used: int
    unused: int
    prorated: Decimal | None  # step 4, unrounded (None for a full refund)
    fee: Decimal | None  # step 4, unrounded (None for a full refund)
    refundable: Decimal  # step 4, USD
    after_prior: Decimal  # step 5, USD
    local: Decimal  # step 6, unrounded
    answer: Decimal  # step 6, rounded


def refund_trace(r: RefundRequest, rounding: str = ROUND_HALF_UP) -> Trace:
    remaining, subtotal = r.seats, Decimal(0)
    for size, rate in TIERS:
        in_tier = remaining if size is None else min(remaining, size)
        subtotal += in_tier * r.list_price_usd * rate
        remaining -= in_tier
    coupon_total = quantize(subtotal, 2, rounding) * (1 - r.coupon_pct / 100)
    paid = quantize(coupon_total, 2, rounding)

    days_used = (r.cancel_date - r.period_start).days + 1
    unused = max(r.period_days - days_used, 0)
    prorated = fee = None
    if r.period_days == 365 and days_used <= 14:
        refundable = paid
    else:
        prorated = paid * unused / r.period_days
        prorated_cents = quantize(prorated, 2, rounding)
        fee = prorated_cents * Decimal("0.03")
        refundable = prorated_cents - quantize(fee, 2, rounding)
    after_prior = max(refundable - r.prior_refund_usd, Decimal(0))
    local = after_prior * r.fx_rate
    answer = quantize(local, MINOR_UNITS[r.currency], rounding)
    return Trace(
        subtotal, coupon_total, paid, days_used, unused, prorated, fee, refundable, after_prior,
        local, answer,
    )  # fmt: skip


def reference_refund(r: RefundRequest) -> Decimal:
    return refund_trace(r).answer
