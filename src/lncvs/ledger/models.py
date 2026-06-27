"""Re-export of the EvidenceLedger type for this module's local namespace.

Per CLAUDE.md, module-level models.py files re-export from schemas/ and must
never define competing types.
"""

from lncvs.schemas import EvidenceLedger

__all__ = ["EvidenceLedger"]
