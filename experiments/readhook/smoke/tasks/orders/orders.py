"""Order bookkeeping helpers for a small warehouse service."""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

VAT_RATES = {"standard": 0.2, "reduced": 0.05, "zero": 0.0}


@dataclass
class Order:
    order_id: int
    sku: str
    qty: int
    unit_price: float
    rate: str = "standard"
    placed: str = "2026-01-01"
    tags: list = field(default_factory=list)


def normalise_sku(sku: str) -> str:
    """SKUs compare case-insensitively, ignoring surrounding spaces and internal hyphens."""
    return sku.strip().upper().replace("-", "")


def line_total(o: Order) -> float:
    """Gross value of one order line, VAT included, rounded to pennies."""
    gross = o.qty * o.unit_price * (1 + VAT_RATES[o.rate])
    return round(gross, 2)


def totals_by_rate(orders: list[Order]) -> dict[str, float]:
    """Sum of line totals per VAT rate; every rate appears, even when zero."""
    out = {r: 0.0 for r in VAT_RATES}
    for o in orders:
        out[o.rate] = round(out[o.rate] + line_total(o), 2)
    return out


def orders_between(orders: list[Order], start: str, end: str) -> list[int]:
    """Ids of orders placed on or after `start` and strictly before `end` (ISO dates), in input order."""
    a, b = dt.date.fromisoformat(start), dt.date.fromisoformat(end)
    return [o.order_id for o in orders if a <= dt.date.fromisoformat(o.placed) < b]


def tag_index(orders: list[Order]) -> dict[str, list[int]]:
    """Map each tag (lower-cased) to the ids of orders carrying it, in input order, without repeats."""
    idx: dict[str, list[int]] = {}
    for o in orders:
        for t in o.tags:
            ids = idx.setdefault(t.lower(), [])
            if o.order_id not in ids:
                ids.append(o.order_id)
    return idx


def dedupe_orders(orders: list[Order]) -> list[Order]:
    """Drop repeated orders.

    Two orders are the same if they share a normalised SKU and the same placed date. The first one is kept,
    except that a later duplicate with a strictly larger qty replaces the kept one in place (the kept order's
    position is preserved). Returns a new list.
    """
    kept: list[Order] = []
    for o in orders:
        key = (normalise_sku(o.sku), o.placed)
        found = None
        for i, k in enumerate(kept):
            if (normalise_sku(k.sku), k.placed) == key:
                found = i
                break
        if found is None:
            kept.append(o)
        elif o.qty > kept[found].qty:
            kept[found] = o
    return kept


def monthly_summary(orders: list[Order]) -> list[tuple[str, int, float]]:
    """(YYYY-MM, number of orders, gross total) per month, months in ascending order."""
    months: dict[str, list[float]] = {}
    for o in orders:
        months.setdefault(o.placed[:7], []).append(line_total(o))
    return [(m, len(v), round(sum(v), 2)) for m, v in sorted(months.items())]
