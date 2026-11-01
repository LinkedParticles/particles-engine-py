#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Complete the engine image's CycloneDX SBOM with what no scanner can see.

The publish-image workflow scans each published platform image with syft, which
catalogs the Debian packages, CPython, and the installed Python closure. It does
not catalog the embedding encoder baked under ``/opt/huggingface``: model
weights carry no package manifest. That encoder is the image's most distinctive
content. The engine pins the revision it fetches (``EMBEDDING_MODEL_REVISION``),
and the SBOM records the revision the image actually holds.

Four subcommands, run in two places:

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

``check-model``
    Runs on the runner, stdlib only, in the job that reads each pushed image
    before any tag is applied. Fails unless the ``baked-model`` output names
    the commit this tree pins in ``encoder-revision.txt`` (security review
    Q40), and unless every file under that commit's snapshot in the image's
    Hugging Face cache, copied out of the image without running it, has the
    SHA-256 ``encoder-files.txt`` records for that commit (security review
    R46). The image's own report names a directory; only the bytes say which
    weights load, so an image that baked other weights is never tagged.

``check``
    Runs on the runner, stdlib only, in the job that signs the SBOM. The
    document was written in a job that ran the image, so before it is signed
    this checks that it names exactly the image identity the signing job
    resolved from the registry itself, and carries one encoder component, at
    the pinned commit.

Edit this in the private upstream's ``publish/overlays/`` — a direct edit on
the public repo is overwritten by the next export.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
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

# The encoder commit this release pins, beside this script. Read from the tree
# rather than from the image, because the image is what is being checked.
PIN_FILE = Path(__file__).with_name("encoder-revision.txt")

# The SHA-256 of each file in the pinned commit, beside the pin, for the same
# reason. The repository is the one the engine loads by its model id.
FILES_FILE = Path(__file__).with_name("encoder-files.txt")
ENCODER_REPO = "sentence-transformers/all-MiniLM-L6-v2"
_FILES_LINE = re.compile(r"([0-9a-f]{64})  (\S(?:.*\S)?)")
# The files a sentence-transformers load reads to decide how the weights are
# used: the module chain, the pooling, the model and tokenizer configs. A bake
# holds only what the load fetched (no ONNX, OpenVINO or other framework
# copies), so the snapshot must hold each of these the pinned commit lists,
# not every file it lists (security review W75).
REQUIRED_FILES = frozenset(
    {
        "modules.json",
        "1_Pooling/config.json",
        "config.json",
        "config_sentence_transformers.json",
        "sentence_bert_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    }
)


def pinned_revision(path: Path = PIN_FILE) -> str:
    """The pinned encoder commit: the one line of ``path`` that is not a comment."""
    lines = [
        ln.strip()
        for ln in path.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.lstrip().startswith("#")
    ]
    if len(lines) != 1 or not re.fullmatch(r"[0-9a-f]{40}", lines[0]):
        raise SystemExit(f"{path} does not hold exactly one 40-hex commit")
    return lines[0]


def pinned_files(path: Path = FILES_FILE) -> dict[str, str]:
    """``relative path -> sha256`` for the pinned commit, from ``path``."""
    files: dict[str, str] = {}
    for ln in path.read_text(encoding="utf-8").splitlines():
        if not ln.strip() or ln.lstrip().startswith("#"):
            continue
        m = _FILES_LINE.fullmatch(ln)
        if not m or m.group(2) in files:
            raise SystemExit(f"{path}: not one `<sha256>  <path>` line per file: {ln!r}")
        files[m.group(2)] = m.group(1)
    if not files:
        raise SystemExit(f"{path} lists no file")
    return files


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def snapshot_problems(hub: Path, *, revision: str, files: dict[str, str]) -> list[str]:
    """Why ``hub`` does not hold exactly the pinned commit's bytes, or an empty list.

    ``hub`` is a copy of the image's Hugging Face hub cache. The pinned
    snapshot must be the repository's only one, each file under it must
    resolve to a regular file inside the repository's cache directory, and
    each must be one ``files`` lists, with that SHA-256. The weights must be
    among them, and so must every `REQUIRED_FILES` entry that ``files`` lists:
    a snapshot without its module chain or pooling config loads the genuine
    weights differently.

    Nothing else in the repository's cache directory may change what a load
    reads (security review R46, W75): the lookup consults ``refs/`` and a
    ``.no_exist/<rev>/<file>`` marker before the snapshot, so the directory
    may hold only ``snapshots/``, ``refs/`` whose every entry names the pin,
    ``blobs/`` holding only the files the snapshot's links reach, and
    ``.no_exist/<pin>/`` markers for files the pinned commit does not hold
    (a load writes those for the optional files it asks for).

    A hub that ``huggingface_hub`` marks as sharing blobs
    (``blobs/.huggingface-shared-blobs``) keeps a large file once, in that
    hub-level store, and the repository's own blob is a link to it: the
    image's bake does this for the weights. A file may then resolve into
    that store instead, and its bytes are checked against the pin the same
    way; anything else outside the repository's directory is still refused.
    """
    name = "models--" + ENCODER_REPO.replace("/", "--")
    repo = hub / name
    if repo.is_symlink() or not repo.is_dir():
        return [f"{hub} holds no {name} directory"]
    tail = "--" + ENCODER_REPO.rsplit("/", 1)[-1]
    others = sorted(p.name for p in hub.iterdir() if p.name.endswith(tail) and p.name != name)
    if others:
        return [f"{hub} holds other caches of the encoder: {others}"]
    snapshots = repo / "snapshots"
    found = sorted(p.name for p in snapshots.iterdir()) if snapshots.is_dir() else []
    if found != [revision]:
        return [f"{repo} holds snapshots {found}, expected only the pin {revision}"]
    snap = snapshots / revision
    root = repo.resolve()
    shared = _shared_store(hub)
    problems: list[str] = []
    seen: list[str] = []
    reached: set[Path] = set()
    for dirpath, dirnames, filenames in os.walk(snap):
        here = Path(dirpath)
        for d in dirnames:
            if (here / d).is_symlink():
                problems.append(f"{(here / d).relative_to(snap)} is a linked directory")
        for f in filenames:
            entry = here / f
            rel = entry.relative_to(snap).as_posix()
            target = entry.resolve()
            inside = target.is_relative_to(root) or (
                shared is not None and target.is_relative_to(shared)
            )
            if not inside or not target.is_file():
                problems.append(f"{rel} does not resolve to a file in the encoder's cache")
                continue
            reached.add(target)
            want = files.get(rel)
            if want is None:
                problems.append(f"{rel} is not a file of the pinned commit")
            elif _sha256(target) != want:
                problems.append(f"{rel} differs from the pinned commit's bytes")
            seen.append(rel)
    if not {"model.safetensors", "pytorch_model.bin"} & set(seen):
        problems.append(f"{snap} holds no weights file")
    for rel in sorted((REQUIRED_FILES & set(files)) - set(seen)):
        problems.append(f"{rel} is missing from the snapshot")
    problems += _cache_problems(
        repo, revision=revision, files=files, reached=reached, shared=shared
    )
    return problems


def _shared_store(hub: Path) -> Path | None:
    """The hub's shared blob store, resolved, when ``huggingface_hub`` marks one."""
    store = hub / "blobs"
    if store.is_symlink() or not (store / ".huggingface-shared-blobs").is_file():
        return None
    return store.resolve()


def _cache_problems(
    repo: Path,
    *,
    revision: str,
    files: dict[str, str],
    reached: set[Path],
    shared: Path | None = None,
) -> list[str]:
    """What in ``repo`` outside its pinned snapshot a load could read instead.

    ``reached`` holds the resolved files the snapshot's entries name; a blob
    may be a link into ``shared``, the hub's shared store, to one of them.
    """
    problems: list[str] = []
    for entry in sorted(repo.iterdir()):
        if entry.name not in {".no_exist", "blobs", "refs", "snapshots"}:
            problems.append(f"{entry.name} in the encoder's cache is not part of the snapshot")
        elif entry.is_symlink() or not entry.is_dir():
            problems.append(f"{entry.name} in the encoder's cache is not a plain directory")
    for sub in (".no_exist", "refs", "blobs"):
        top = repo / sub
        if top.is_symlink() or not top.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(top):
            here = Path(dirpath)
            for d in dirnames:
                if (here / d).is_symlink():
                    problems.append(f"{(here / d).relative_to(repo)} is a linked directory")
            for f in filenames:
                path = here / f
                rel = path.relative_to(repo).as_posix()
                if (
                    sub == "blobs"
                    and path.is_symlink()
                    and shared is not None
                    and path.resolve().is_relative_to(shared)
                ):
                    if path.resolve() not in reached:
                        problems.append(f"{rel} is not a file of the pinned snapshot")
                elif path.is_symlink() or not path.is_file():
                    problems.append(f"{rel} is not a regular file")
                elif sub == ".no_exist":
                    rev, _, marked = path.relative_to(top).as_posix().partition("/")
                    if rev != revision:
                        problems.append(f"{rel} marks a revision other than the pin")
                    elif marked in files:
                        problems.append(f"{rel} hides a file of the pinned commit from a load")
                elif sub == "refs":
                    if path.read_text(encoding="utf-8", errors="replace").strip() != revision:
                        problems.append(f"{rel} names a revision other than the pin")
                elif path.resolve() not in reached:
                    problems.append(f"{rel} is not a file of the pinned snapshot")
    return problems


def check_model(
    model: dict[str, Any],
    *,
    hub: Path | None = None,
    revision: str | None = None,
    files: dict[str, str] | None = None,
) -> list[str]:
    """Why a baked encoder is not the pinned one, or an empty list.

    ``model`` is the ``baked-model`` output, which the image wrote about
    itself; ``hub`` is the image's hub cache, copied out of it. With no
    ``hub`` the check fails, since a name alone proves nothing.
    """
    pin = revision or pinned_revision()
    commit = str(model.get("commit", "")).lower()
    if commit != pin:
        return [f"the image baked {model.get('repo_id')}@{commit or '?'}, but the pin is {pin}"]
    if model.get("repo_id") != ENCODER_REPO:
        return [f"the image baked {model.get('repo_id')}, not {ENCODER_REPO}"]
    if hub is None:
        return ["no copy of the image's hub cache was given, so no baked byte was compared"]
    return snapshot_problems(hub, revision=pin, files=files or pinned_files())


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
    # A release that pins the revision must have baked that one. Older
    # releases name no pin, and an SBOM backfill still describes them.
    from particles import embeddings

    pinned = getattr(embeddings, "EMBEDDING_MODEL_REVISION", None)
    if pinned and rev.commit_hash != pinned:
        raise SystemExit(f"baked {repo.repo_id}@{rev.commit_hash}, but the engine pins {pinned}")
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


def check(
    bom: dict[str, Any],
    *,
    image: str,
    tag: str,
    arch: str,
    platform_digest: str,
    index_digest: str,
    encoder_revision: str | None = None,
) -> list[str]:
    """Why ``bom`` is not the enriched SBOM of this image, or an empty list."""
    if bom.get("bomFormat") != "CycloneDX" or bom.get("specVersion") != "1.6":
        return [f"not a CycloneDX 1.6 document: {bom.get('bomFormat')} {bom.get('specVersion')}"]
    problems = []
    metadata = bom.get("metadata") or {}
    props: dict[str, list[str]] = {}
    for prop in metadata.get("properties") or []:
        props.setdefault(str(prop.get("name")), []).append(str(prop.get("value")))
    expected = {
        f"{PROP_PREFIX}:tag": f"{image}:{tag}",
        f"{PROP_PREFIX}:platform": f"linux/{arch}",
        f"{PROP_PREFIX}:platform-digest": platform_digest,
        f"{PROP_PREFIX}:index-digest": index_digest,
    }
    for name, want in expected.items():
        if props.get(name) != [want]:
            problems.append(f"{name} is {props.get(name)}, expected [{want!r}]")
    purl = (metadata.get("component") or {}).get("purl")
    if purl != image_purl(image, platform_digest, arch):
        problems.append(f"the subject purl {purl!r} names another image")
    models = [c for c in bom.get("components") or [] if c.get("type") == "machine-learning-model"]
    if len(models) != 1:
        problems.append(f"expected one encoder component, found {len(models)}")
    else:
        pin = encoder_revision or pinned_revision()
        named = models[0].get("version")
        if str(named or "").lower() != pin:
            problems.append(f"the encoder component names commit {named!r}, not the pin {pin}")
    return problems


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
    cm = sub.add_parser("check-model", help="check the baked encoder is the pinned commit's bytes")
    cm.add_argument("model", type=Path)
    cm.add_argument(
        "--hub", type=Path, required=True, help="the image's hub cache, copied out of it"
    )
    ck = sub.add_parser("check", help="check an enriched SBOM names this image before signing it")
    ck.add_argument("sbom", type=Path)
    for flag in ("--image", "--tag", "--arch", "--platform-digest", "--index-digest"):
        ck.add_argument(flag, required=True)
    args = parser.parse_args(argv)

    if args.cmd == "baked-model":
        print(json.dumps(baked_model()))
        return 0
    if args.cmd == "check-model":
        problems = check_model(json.loads(args.model.read_text(encoding="utf-8")), hub=args.hub)
        for problem in problems:
            print(f"{args.model}: {problem}", file=sys.stderr)
        return 1 if problems else 0
    if args.cmd == "check":
        if args.sbom.stat().st_size > ATTEST_MAX_BYTES:
            print(f"{args.sbom} is over the attestation cap", file=sys.stderr)
            return 1
        problems = check(
            json.loads(args.sbom.read_text(encoding="utf-8")),
            image=args.image,
            tag=args.tag,
            arch=args.arch,
            platform_digest=args.platform_digest,
            index_digest=args.index_digest,
        )
        for problem in problems:
            print(f"{args.sbom}: {problem}", file=sys.stderr)
        return 1 if problems else 0

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
