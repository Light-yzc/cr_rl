"""Split the result bundle for bounded, checksummed Colab CLI downloads."""

import hashlib
import json
from pathlib import Path

archive = Path("/content/colab_smoke_results.tar.gz")
parts = []
with archive.open("rb") as stream:
    while data := stream.read(16 * 1024 * 1024):
        part = archive.with_name(f"colab_results.part{len(parts):04d}")
        part.write_bytes(data)
        parts.append(
            {"name": part.name, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        )
manifest = {"size": archive.stat().st_size, "parts": parts}
archive.with_name("colab_results_manifest.json").write_text(json.dumps(manifest))
print("Result bundle bytes:", manifest["size"], "parts:", len(parts), flush=True)
