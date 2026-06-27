"""last_write_wins reducer tests."""

from lncvs.orchestration.reducers import last_write_wins


def test_last_write_wins_returns_the_update() -> None:
    assert last_write_wins("old", "new") == "new"


def test_last_write_wins_ignores_current_value_entirely() -> None:
    assert last_write_wins(current=999, update=1) == 1


def test_last_write_wins_works_for_arbitrary_objects() -> None:
    class _Thing:
        def __init__(self, value: int) -> None:
            self.value = value

    old, new = _Thing(1), _Thing(2)
    assert last_write_wins(old, new) is new
