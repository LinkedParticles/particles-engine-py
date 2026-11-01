# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0
"""Give a reproducible CycloneDX SBOM the serial number an attestation needs.

``actions/attest`` reads a document as CycloneDX only when it carries
``bomFormat``, ``specVersion`` and ``serialNumber``, and the generator's
``--output-reproducible`` drops the random serial so that a re-run is
byte-identical. The serial written here is a UUIDv5 of the document's own
SHA-256, so it is a valid ``urn:uuid:`` and the file stays reproducible.

Standard library only: the sbom job runs it with the hash-locked generator's
interpreter in isolated mode, so no code from the inventoried closure runs.
Usage: ``python -I sbom_serial.py <sbom.cdx.json>``, in place.
"""

from __future__ import annotations

import hashlib
import json
import sys
import uuid
from pathlib import Path
from typing import Any


def with_serial(raw: bytes) -> dict[str, Any]:
    """The document in ``raw`` with a serial derived from its bytes."""
    bom: dict[str, Any] = json.loads(raw)
    if bom.get("bomFormat") != "CycloneDX" or not bom.get("specVersion"):
        raise ValueError("not a CycloneDX document")
    if "serialNumber" in bom:
        raise ValueError("the document already has a serialNumber")
    digest = hashlib.sha256(raw).hexdigest()
    bom["serialNumber"] = f"urn:uuid:{uuid.uuid5(uuid.NAMESPACE_URL, f'sha256:{digest}')}"
    return bom


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print("usage: sbom_serial.py <sbom.cdx.json>", file=sys.stderr)
        return 2
    path = Path(argv[0])
    try:
        bom = with_serial(path.read_bytes())
    except ValueError as exc:
        print(f"{path}: {exc}", file=sys.stderr)
        return 1
    path.write_text(json.dumps(bom, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{path}: serialNumber {bom['serialNumber']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
