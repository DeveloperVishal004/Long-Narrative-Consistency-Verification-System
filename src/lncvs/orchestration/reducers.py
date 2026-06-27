"""Explicit reducer functions for LangGraph state channels.

Per the Phase 7 architecture review's reducer reconciliation:
LedgerService's append-only mutation methods ARE the effective reducer
semantics, applied at the EvidenceLedger-object level. At the LangGraph
channel level, `ledger` and `control` are whole-object channels using
last-write-wins -- valid because this graph is strictly linear with no
parallel fan-out, so no two nodes ever write to the same channel within the
same super-step. CLAUDE.md's "never rely on default overwrite behavior for
list fields" is honored by declaring this explicitly via Annotated, not by
relying on LangGraph's implicit default reducer.

If a later phase ever adds parallel fan-out (e.g. per-claim parallel
retrieval), this reducer choice must be revisited -- it is correct only
because Phase 7 is a faithful, strictly linear port.
"""

from typing import TypeVar

T = TypeVar("T")


def last_write_wins(current: T, update: T) -> T:
    """The new value always wins.

    Valid only because no two nodes in this graph ever write to the same
    channel within the same super-step -- see module docstring.
    """
    return update
