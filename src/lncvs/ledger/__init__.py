"""Evidence Ledger mutation API.

The EvidenceLedger model itself lives in lncvs.schemas.ledger. This package
provides the only sanctioned way to mutate one: lncvs.ledger.service.LedgerService.
"""

from lncvs.ledger.service import LedgerService

__all__ = ["LedgerService"]
