"""Independent implementation of the refund policy in `helphub.refunds.plan.REFUND_POLICY`.

Used only to check that the expected answers stored in tests/fixtures/refunds.json are correct;
the application itself delegates this calculation to the model.
"""

from decimal import ROUND_HALF_UP, Decimal

from helphub.refunds.plan import RefundRequest

TIERS = [(10, Decimal("1.00")), (40, Decimal("0.85")), (None, Decimal("0.70"))]  # (seats, rate)
MINOR_UNITS = {"USD": 2, "EUR": 2, "GBP": 2, "JPY": 0}


def round_half_up(value: Decimal, places: int) -> Decimal:
    return value.quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP)


def reference_refund(r: RefundRequest) -> Decimal:
    remaining, subtotal = r.seats, Decimal(0)
    for size, rate in TIERS:
        in_tier = remaining if size is None else min(remaining, size)
        subtotal += in_tier * r.list_price_usd * rate
        remaining -= in_tier
    paid = round_half_up(subtotal, 2)
    paid = round_half_up(paid * (1 - r.coupon_pct / 100), 2)

    days_used = (r.cancel_date - r.period_start).days + 1
    unused = max(r.period_days - days_used, 0)
    if r.period_days == 365 and days_used <= 14:
        refundable = paid
    else:
        prorated = round_half_up(paid * unused / r.period_days, 2)
        refundable = prorated - round_half_up(prorated * Decimal("0.03"), 2)
    refundable = max(refundable - r.prior_refund_usd, Decimal(0))
    return round_half_up(refundable * r.fx_rate, MINOR_UNITS[r.currency])
