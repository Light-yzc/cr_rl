"""Apply and identify the small, pinned crforge command-lookup correction.

The upstream revision alone cannot identify a locally patched build. The patch
manifest identifies source, while a build receipt hashes the actual installed
jars. A JVM marker also detects a process that loaded an older core jar.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

from cr_rl.config import ROOT
from cr_rl.training.persistence import atomic_json, file_sha256

RECEIPT = "cr_rl_backend.json"


def patch_manifest(root: Path = ROOT):
    import yaml

    config = root / "configs/backends.yaml"
    relative = (
        yaml.safe_load(config.read_text())["primary"].get(
            "patch_manifest", "patches/crforge/manifest.json"
        )
        if config.exists()
        else "patches/crforge/manifest.json"
    )
    path = root / relative
    manifest = json.loads(path.read_text(encoding="utf-8"))
    for patch in manifest["patches"]:
        if file_sha256(root / patch["path"]) != patch["sha256"]:
            raise ValueError(f"Simulator patch checksum mismatch: {patch['path']}")
    return manifest, file_sha256(path)


def _git(vendor, *args, check=True):
    return subprocess.run(
        ["git", "-C", str(vendor), *args], check=check, capture_output=True, text=True
    )


def _source_state(vendor, manifest):
    result = {}
    for name, expected in manifest["files"].items():
        path = vendor / name
        digest = file_sha256(path) if path.is_file() else None
        if digest not in (expected["base_sha256"], expected["patched_sha256"]):
            raise ValueError(f"Simulator source has unrelated edits: {name}")
        result[name] = "patched" if digest == expected["patched_sha256"] else "base"
    return result


def apply_patches(vendor: Path, *, root: Path = ROOT):
    manifest, manifest_sha = patch_manifest(root)
    commit = _git(vendor, "rev-parse", "HEAD").stdout.strip()
    if commit != manifest["base_commit"]:
        raise ValueError(f"Simulator patch requires pinned revision {manifest['base_commit']}")
    states = set(_source_state(vendor, manifest).values())
    if states == {"patched"}:
        return {"applied": False, "manifest_sha256": manifest_sha}
    if states != {"base"}:
        raise ValueError(
            "Simulator patch is only partly applied; preserve edits and inspect source"
        )
    # Validate the complete series first. One patch currently contains the fix
    # and its native tests; git apply writes the working tree, never the index.
    patch_paths = [str((root / item["path"]).resolve()) for item in manifest["patches"]]
    _git(vendor, "apply", "--check", *patch_paths)
    _git(vendor, "apply", *patch_paths)
    if set(_source_state(vendor, manifest).values()) != {"patched"}:
        raise ValueError("Applied simulator patch does not match its declared source")
    return {"applied": True, "manifest_sha256": manifest_sha}


def jar_fingerprints(install: Path):
    jars = sorted((install / "lib").glob("*.jar"))
    if not jars:
        raise FileNotFoundError(f"No simulator jars installed under {install}")
    return {path.name: file_sha256(path) for path in jars}


def write_build_receipt(vendor: Path, install: Path, *, root: Path = ROOT):
    manifest, manifest_sha = patch_manifest(root)
    commit = _git(vendor, "rev-parse", "HEAD").stdout.strip()
    if commit != manifest["base_commit"]:
        raise ValueError("Cannot publish a build receipt for an unexpected simulator revision")
    if set(_source_state(vendor, manifest).values()) != {"patched"}:
        raise ValueError("Simulator sources are not the declared patched sources")
    cores = list((install / "lib").glob("core-*.jar"))
    if len(cores) != 1:
        raise ValueError("Build receipt requires exactly one installed core jar")
    with zipfile.ZipFile(cores[0]) as archive:
        code = archive.read("org/crforge/core/battle/unit/Standard1v1Battle.class")
    if manifest["runtime_marker"].encode() not in code or b"commandLookupPatch" not in code:
        raise ValueError("Installed core jar does not contain the registered patch marker")
    diff = _git(vendor, "diff", "--binary", "HEAD").stdout.encode()
    receipt = {
        "schema_version": 1,
        "base_commit": commit,
        "patch_manifest_sha256": manifest_sha,
        "runtime_marker": manifest["runtime_marker"],
        "patches": manifest["patches"],
        "source_sha256": {
            name: value["patched_sha256"] for name, value in manifest["files"].items()
        },
        "jar_sha256": jar_fingerprints(install),
        "vendor_diff_sha256": hashlib.sha256(diff).hexdigest(),
    }
    atomic_json(install / RECEIPT, receipt)
    return receipt


def installed_backend(install: Path, *, verify=True, require_patch=True, root: Path = ROOT):
    """Read actual installed provenance; never infer patching from Git HEAD."""
    path = install / RECEIPT
    if not path.exists():
        if require_patch:
            raise ValueError(
                "Simulator patch build receipt is missing. Run scripts/setup_simulator.py "
                "to apply and build the pinned late-champion fix."
            )
        return {
            "patch_manifest_sha256": None,
            "patches": [],
            "jar_sha256": jar_fingerprints(install),
            "verified_patch": False,
        }
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if verify:
        manifest, manifest_sha = patch_manifest(root)
        if (
            receipt.get("schema_version") != 1
            or receipt.get("base_commit") != manifest["base_commit"]
            or receipt.get("patch_manifest_sha256") != manifest_sha
            or receipt.get("runtime_marker") != manifest["runtime_marker"]
            or receipt.get("patches") != manifest["patches"]
        ):
            raise ValueError("Simulator build receipt differs from the registered patch manifest")
        if receipt.get("jar_sha256") != jar_fingerprints(install):
            raise ValueError("Installed simulator jars changed since their build receipt")
    return {**receipt, "verified_patch": bool(verify)}


def loaded_backend(install: Path, *, root: Path = ROOT):
    """Check the active JVM, after VectorBattle has started it."""
    info = installed_backend(install, root=root)
    import jpype

    core = jpype.JClass("org.crforge.core.battle.unit.Standard1v1Battle")
    try:
        marker = str(core.commandLookupPatch())
    except AttributeError as error:
        raise ValueError(
            "The JVM loaded an unpatched core jar; rebuild and use a fresh process"
        ) from error
    if marker != info["runtime_marker"]:
        raise ValueError("Loaded simulator marker differs from the registered patch")
    return info
