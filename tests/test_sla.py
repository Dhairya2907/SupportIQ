from datetime import datetime, timedelta, timezone

from app.services import sla

IST = sla.IST


def ist(y, m, d, h, mi=0):
    return datetime(y, m, d, h, mi, tzinfo=IST)


def test_business_hours_within_same_day():
    # Wednesday 10:00 IST + 2h -> 12:00 IST
    start = ist(2026, 11, 11, 10)
    assert sla.add_business_hours(start, 2).astimezone(IST) == ist(2026, 11, 11, 12)


def test_business_hours_roll_over_close_and_sunday():
    # Saturday 20:00 IST + 2h -> 1h today (to 21:00) + 1h Monday (09:00 -> 10:00)
    start = ist(2026, 11, 14, 20)  # 2026-11-14 is a Saturday
    assert sla.add_business_hours(start, 2).astimezone(IST) == ist(2026, 11, 16, 10)


def test_before_opening_starts_at_nine():
    start = ist(2026, 11, 11, 6)
    assert sla.add_business_hours(start, 1).astimezone(IST) == ist(2026, 11, 11, 10)


def test_business_days_skip_sunday():
    # Saturday + 1 business day -> Monday 21:00
    start = ist(2026, 11, 14, 12)
    assert sla.add_business_days(start, 1).astimezone(IST) == ist(2026, 11, 16, 21)


def test_deadlines_by_priority_order():
    created = ist(2026, 11, 11, 10)
    r_high, s_high = sla.deadlines(created, "high")
    r_low, s_low = sla.deadlines(created, "low")
    assert r_high < r_low and s_high < s_low


def test_evaluate_states():
    now = datetime(2026, 11, 11, 12, 0, tzinfo=timezone.utc)
    due_far = now + timedelta(hours=5)
    due_soon = now + timedelta(minutes=30)
    due_past = now - timedelta(minutes=1)
    assert sla.evaluate(due_far, None, "response", now)["status"] == "on_track"
    assert sla.evaluate(due_soon, None, "response", now)["status"] == "at_risk"
    assert sla.evaluate(due_past, None, "response", now)["status"] == "breached"
    assert sla.evaluate(due_far, now, "response", now)["status"] == "met"
    assert sla.evaluate(due_past, now, "response", now)["status"] == "breached"
    assert sla.evaluate(None, None, "response", now) is None


def test_naive_datetimes_are_treated_as_utc():
    naive = datetime(2026, 11, 11, 4, 30)  # 10:00 IST
    assert sla.add_business_hours(naive, 1).astimezone(IST) == ist(2026, 11, 11, 11)
