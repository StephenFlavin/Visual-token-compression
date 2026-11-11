from orders import Order, dedupe_orders, totals_by_rate


def test_dedupe_keeps_first_and_position():
    a = Order(1, "ab-1", 2, 1.0)
    b = Order(2, "XY", 1, 1.0)
    c = Order(3, " AB1 ", 5, 1.0)
    out = dedupe_orders([a, b, c])
    assert [o.order_id for o in out] == [3, 2]


def test_dedupe_equal_qty_keeps_first():
    a, b = Order(1, "q", 2, 1.0), Order(2, "Q", 2, 1.0)
    assert [o.order_id for o in dedupe_orders([a, b])] == [1]


def test_totals():
    assert totals_by_rate([Order(1, "a", 1, 10.0)])["standard"] == 12.0
