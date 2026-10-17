from api.limits import Limits, visitor_id


class Clock:
    def __init__(self, t=1_800_000_000.0): self.t = t
    def __call__(self): return self.t


def test_limits_off_by_default():
    lim = Limits(per_visitor=0, llm_per_day=0)
    assert all(lim.visitor_wait_s("a") == 0 for _ in range(100))
    assert all(lim.take_llm_call() for _ in range(1000))


def test_per_visitor_limit_rolls_over_after_an_hour():
    clock = Clock(); lim = Limits(per_visitor=3, llm_per_day=0, clock=clock)
    assert [lim.visitor_wait_s("a") for _ in range(3)] == [0, 0, 0]
    assert lim.visitor_wait_s("a") > 0          # 4th question within the hour is refused
    assert lim.visitor_wait_s("b") == 0         # other visitors unaffected
    clock.t += 3600
    assert lim.visitor_wait_s("a") == 0


def test_refused_asks_do_not_count():
    clock = Clock(); lim = Limits(per_visitor=1, llm_per_day=0, clock=clock)
    lim.visitor_wait_s("a")
    for _ in range(5): lim.visitor_wait_s("a")  # refused, must not extend the wait
    clock.t += 3600
    assert lim.visitor_wait_s("a") == 0


def test_daily_llm_budget_resets_at_utc_midnight():
    clock = Clock(1_800_000_000.0); lim = Limits(per_visitor=0, llm_per_day=2, clock=clock)
    assert [lim.take_llm_call() for _ in range(3)] == [True, True, False]
    clock.t += 86400
    assert lim.take_llm_call() is True


def test_visitor_id_uses_the_proxy_added_address():
    assert visitor_id({"x-forwarded-for": "6.6.6.6, 1.2.3.4"}, "10.0.0.1") == "1.2.3.4"
    assert visitor_id({}, "10.0.0.1") == "10.0.0.1"
