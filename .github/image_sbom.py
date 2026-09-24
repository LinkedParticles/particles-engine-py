#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Complete the engine image's CycloneDX SBOM with what no scanner can see.

The publish-image workflow scans each published platform image with syft, which
catalogs the Debian packages, CPython, and the installed Python closure. It does
not catalog the embedding encoder baked under ``/opt/huggingface``: model
weights carry no package manifest. That encoder is the image's most distinctive
content, and its revision is recorded nowhere else — the code names the model
but not a revision, so each build bakes whatever the hub's ``main`` pointed at.

Two subcommands, run in two places:

``baked-model``
    Runs **inside the image** (``docker run … python image_sbom.py
    baked-model``). Finds the one cached model matching the engine's
    ``EMBEDDING_MODEL_ID`` and prints its repo id, resolved commit, and the
    licence its model card declares, as one JSON object.

``enrich``
    Runs on the runner, stdlib only. Adds that model to the syft SBOM as a
    CycloneDX ``machine-learning-model`` component, stamps the image identity
    (tag, platform digest, manifest-list digest) onto the metadata, and fails
    if the document is not CycloneDX 1.6 or exceeds the attestation size cap.

Edit this in the private upstream's ``publish/overlays/`` — a direct edit on
the public repo is overwritten by the next export.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any
from urllib.parse import quote

# The signed-attestation action refuses an SBOM larger than this.
ATTEST_MAX_BYTES = 16 * 1024 * 1024

# Hugging Face model cards use lowercase licence keys; map the ones that are
# SPDX ids so the component carries a machine-comparable licence.
_HF_TO_SPDX = {
    "apache-2.0": "Apache-2.0",
    "mit": "MIT",
    "bsd-3-clause": "BSD-3-Clause",
    "cc-by-4.0": "CC-BY-4.0",
}

PROP_PREFIX = "linkedparticles:image"


def baked_model() -> dict[str, Any]:
    """Describe the encoder baked into this image (run inside the image)."""
    from huggingface_hub import scan_cache_dir

    from particles.embeddings import EMBEDDING_MODEL_ID

    models = [r for r in scan_cache_dir().repos if r.repo_type == "model"]
    match = [r for r in models if r.repo_id.rsplit("/", 1)[-1] == EMBEDDING_MODEL_ID]
    if len(match) != 1:
        raise SystemExit(
            f"expected exactly one cached {EMBEDDING_MODEL_ID!r}, "
            f"found {[r.repo_id for r in models]}"
        )
    (repo,) = match
    if len(repo.revisions) != 1:
        # A fresh bake holds one snapshot; more means the cache was reused and
        # "which weights load" is no longer a single answer.
        raise SystemExit(
            f"expected one baked revision of {repo.repo_id}, found {len(repo.revisions)}"
        )
    (rev,) = repo.revisions
    return {
        "repo_id": repo.repo_id,
        "commit": rev.commit_hash,
        "license": card_license(Path(rev.snapshot_path) / "README.md"),
        "path": str(repo.repo_path),
    }


def card_license(card: Path) -> str | None:
    """The ``license:`` key from a model card's YAML frontmatter, if any."""
    if not card.is_file():
        return None
    text = card.read_text(encoding="utf-8")
    if not text.startswith("---"):
        return None
    parts = text.split("---", 2)
    if len(parts) < 3:
        return None
    for line in parts[1].splitlines():
        if line.startswith("license:"):
            return line.split(":", 1)[1].strip().strip("'\"") or None
    return None


def model_component(model: dict[str, Any]) -> dict[str, Any]:
    """The CycloneDX component for the baked encoder."""
    namespace, _, name = model["repo_id"].rpartition("/")
    commit = model["commit"].lower()
    path = f"{namespace}/{name}" if namespace else name
    purl = f"pkg:huggingface/{path}@{commit}"
    component: dict[str, Any] = {
        "type": "machine-learning-model",
        "bom-ref": purl,
        "name": name,
        "version": commit,
        "purl": purl,
        "externalReferences": [
            {
                "type": "distribution",
                "url": f"https://huggingface.co/{model['repo_id']}/tree/{commit}",
            }
        ],
        "properties": [{"name": f"{PROP_PREFIX}:baked-path", "value": model["path"]}],
    }
    if namespace:
        component["group"] = namespace
    lic = model.get("license")
    if lic:
        spdx = _HF_TO_SPDX.get(lic.lower())
        component["licenses"] = [{"license": {"id": spdx} if spdx else {"name": lic}}]
    return component


def image_purl(image: str, digest: str, arch: str) -> str:
    """``pkg:oci`` purl for one platform image (``ghcr.io/org/name`` form)."""
    repository, _, name = image.rpartition("/")
    return (
        f"pkg:oci/{name}@{quote(digest, safe='')}"
        f"?arch={arch}&repository_url={quote(f'{repository}/{name}', safe='/')}"
    )


def enrich(
    bom: dict[str, Any],
    *,
    model: dict[str, Any],
    image: str,
    tag: str,
    arch: str,
    platform_digest: str,
    index_digest: str,
) -> dict[str, Any]:
    """Return ``bom`` with the encoder component and image identity added."""
    if bom.get("bomFormat") != "CycloneDX" or bom.get("specVersion") != "1.6":
        raise ValueError(
            "expected a CycloneDX 1.6 document, "
            f"got {bom.get('bomFormat')} {bom.get('specVersion')}"
        )

    component = model_component(model)
    components = bom.setdefault("components", [])
    if any(c.get("bom-ref") == component["bom-ref"] for c in components):
        raise ValueError(f"{component['bom-ref']} is already in the SBOM")
    components.append(component)

    metadata = bom.setdefault("metadata", {})
    subject = metadata.setdefault("component", {"type": "container", "name": image})
    subject.setdefault("bom-ref", f"{image}@{platform_digest}")
    subject["purl"] = image_purl(image, platform_digest, arch)
    props = metadata.setdefault("properties", [])
    props.extend(
        [
            {"name": f"{PROP_PREFIX}:tag", "value": f"{image}:{tag}"},
            {"name": f"{PROP_PREFIX}:platform", "value": f"linux/{arch}"},
            {"name": f"{PROP_PREFIX}:platform-digest", "value": platform_digest},
            {"name": f"{PROP_PREFIX}:index-digest", "value": index_digest},
        ]
    )

    # The image depends on the encoder it ships.
    deps = bom.setdefault("dependencies", [])
    root = next((d for d in deps if d.get("ref") == subject["bom-ref"]), None)
    if root is None:
        root = {"ref": subject["bom-ref"], "dependsOn": []}
        deps.append(root)
    root.setdefault("dependsOn", []).append(component["bom-ref"])
    return bom


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("baked-model", help="describe the baked encoder (run inside the image)")
    en = sub.add_parser("enrich", help="add the encoder + image identity to a syft SBOM, in place")
    en.add_argument("sbom", type=Path)
    en.add_argument("--model", type=Path, required=True, help="baked-model JSON output")
    en.add_argument("--image", required=True)
    en.add_argument("--tag", required=True)
    en.add_argument("--arch", required=True)
    en.add_argument("--platform-digest", required=True)
    en.add_argument("--index-digest", required=True)
    args = parser.parse_args(argv)

    if args.cmd == "baked-model":
        print(json.dumps(baked_model()))
        return 0

    bom = enrich(
        json.loads(args.sbom.read_text(encoding="utf-8")),
        model=json.loads(args.model.read_text(encoding="utf-8")),
        image=args.image,
        tag=args.tag,
        arch=args.arch,
        platform_digest=args.platform_digest,
        index_digest=args.index_digest,
    )
    out = json.dumps(bom, indent=2) + "\n"
    size = len(out.encode("utf-8"))
    if size > ATTEST_MAX_BYTES:
        cap = ATTEST_MAX_BYTES
        print(f"SBOM is {size} bytes, over the {cap}-byte attestation cap", file=sys.stderr)
        return 1
    args.sbom.write_text(out, encoding="utf-8")
    print(f"enriched {args.sbom} ({size} bytes): + {bom['components'][-1]['purl']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
