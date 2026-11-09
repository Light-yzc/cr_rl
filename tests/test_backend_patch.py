"""Prevent silent use of an old or modified simulator after a source patch."""

import hashlib
import json
import subprocess
import zipfile

import pytest

from cr_rl.envs.backend_patch import apply_patches, installed_backend, write_build_receipt


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(vendor, *args):
    return subprocess.run(
        ["git", "-C", str(vendor), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def fixture(tmp_path):
    vendor, root, install = (tmp_path / name for name in ("vendor", "root", "install"))
    vendor.mkdir()
    run(vendor, "init")
    run(vendor, "config", "user.name", "Backend fixture")
    run(vendor, "config", "user.email", "fixture@example.invalid")
    source = vendor / "core/Example.java"
    source.parent.mkdir()
    source.write_text("before\n")
    run(vendor, "add", "core/Example.java")
    run(vendor, "commit", "-m", "Local fixture")
    commit = run(vendor, "rev-parse", "HEAD")
    patches = root / "patches/crforge"
    patches.mkdir(parents=True)
    patch = patches / "fix.patch"
    patch.write_text(
        "--- a/core/Example.java\n+++ b/core/Example.java\n@@ -1 +1 @@\n-before\n+after\n"
    )
    manifest = {
        "base_commit": commit,
        "runtime_marker": "test_marker_v1",
        "patches": [{"id": "test", "path": "patches/crforge/fix.patch", "sha256": digest(patch)}],
        "files": {
            "core/Example.java": {
                "base_sha256": digest(source),
                "patched_sha256": hashlib.sha256(b"after\n").hexdigest(),
            }
        },
    }
    (patches / "manifest.json").write_text(json.dumps(manifest))
    (install / "lib").mkdir(parents=True)
    with zipfile.ZipFile(install / "lib/core-test.jar", "w") as archive:
        # Contract probe only: this is not presented as compiled Java bytecode.
        archive.writestr(
            "org/crforge/core/battle/unit/Standard1v1Battle.class",
            b"test_marker_v1 commandLookupPatch",
        )
    (install / "lib/env-test.jar").write_bytes(b"env fixture")
    return vendor, root, install


def test_apply_is_idempotent_and_preserves_unrelated_edits(tmp_path):
    vendor, root, _ = fixture(tmp_path)
    assert apply_patches(vendor, root=root)["applied"]
    assert (vendor / "core/Example.java").read_text() == "after\n"
    assert not apply_patches(vendor, root=root)["applied"]
    (vendor / "core/Example.java").write_text("user changes\n")
    with pytest.raises(ValueError, match="unrelated edits"):
        apply_patches(vendor, root=root)
    assert (vendor / "core/Example.java").read_text() == "user changes\n"


def test_patch_checksum_and_wrong_revision_fail_before_mutation(tmp_path):
    vendor, root, _ = fixture(tmp_path)
    patch = root / "patches/crforge/fix.patch"
    patch.write_text(patch.read_text() + "tampered\n")
    with pytest.raises(ValueError, match="checksum"):
        apply_patches(vendor, root=root)
    assert (vendor / "core/Example.java").read_text() == "before\n"


def test_receipt_verifies_actual_jars_and_missing_receipt_is_explicit(tmp_path):
    vendor, root, install = fixture(tmp_path)
    with pytest.raises(ValueError, match="receipt is missing"):
        installed_backend(install, root=root)
    assert not installed_backend(install, root=root, require_patch=False)["verified_patch"]
    apply_patches(vendor, root=root)
    record = write_build_receipt(vendor, install, root=root)
    assert installed_backend(install, root=root)["jar_sha256"] == record["jar_sha256"]
    (install / "lib/env-test.jar").write_bytes(b"replaced after build")
    with pytest.raises(ValueError, match="jars changed"):
        installed_backend(install, root=root)


def test_source_patch_alone_cannot_publish_an_old_core_jar(tmp_path):
    vendor, root, install = fixture(tmp_path)
    apply_patches(vendor, root=root)
    with zipfile.ZipFile(install / "lib/core-test.jar", "w") as archive:
        archive.writestr("org/crforge/core/battle/unit/Standard1v1Battle.class", b"old core")
    with pytest.raises(ValueError, match="patch marker"):
        write_build_receipt(vendor, install, root=root)


def test_receipt_cannot_claim_different_patch_manifest(tmp_path):
    vendor, root, install = fixture(tmp_path)
    apply_patches(vendor, root=root)
    write_build_receipt(vendor, install, root=root)
    manifest_path = root / "patches/crforge/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["runtime_marker"] = "another_patch"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="receipt differs"):
        installed_backend(install, root=root)


def test_other_revision_is_rejected_before_applying_patch(tmp_path):
    vendor, root, _ = fixture(tmp_path)
    run(vendor, "commit", "--allow-empty", "-m", "Different revision")
    with pytest.raises(ValueError, match="pinned revision"):
        apply_patches(vendor, root=root)
    assert (vendor / "core/Example.java").read_text() == "before\n"
