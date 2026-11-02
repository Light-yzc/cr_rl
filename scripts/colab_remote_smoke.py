from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path("/content/cr-rl-lab")
ARCHIVE = Path("/content/cr-rl-lab-src.tar.gz")
COMMIT = "7ee59b39bf1d41d5c01645f01c47ed8c8d3355d6"


def run(command: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(map(str, command)), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def main() -> None:
    if not ARCHIVE.exists():
        raise FileNotFoundError(f"Upload the source archive first: {ARCHIVE}")
    ROOT.mkdir(parents=True, exist_ok=True)
    with tarfile.open(ARCHIVE, "r:gz") as archive:
        archive.extractall(ROOT, filter="data")

    print("=== Colab runtime ===", flush=True)
    run(
        [
            sys.executable,
            "-c",
            "import torch\n"
            "print(torch.__version__, torch.cuda.is_available())\n"
            "print(torch.cuda.get_device_name(0))\n"
            "print(round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2), 'GiB')",
        ]
    )
    run(
        [
            "bash",
            "-lc",
            "apt-get update -qq && DEBIAN_FRONTEND=noninteractive "
            "apt-get install -y -qq openjdk-21-jdk-headless git",
        ]
    )
    java_home = "/usr/lib/jvm/java-21-openjdk-amd64"
    os.environ["JAVA_HOME"] = java_home
    os.environ["PATH"] = f"{java_home}/bin:" + os.environ["PATH"]
    run(["java", "-version"])

    vendor = ROOT / ".cache/vendor/crforge"
    vendor.parent.mkdir(parents=True, exist_ok=True)
    if not (vendor / ".git").exists():
        run(["git", "clone", "https://github.com/voonhous/crforge.git", str(vendor)])
    run(["git", "fetch", "--depth", "1", "origin", COMMIT], cwd=vendor)
    run(["git", "checkout", "--detach", COMMIT], cwd=vendor)
    run(
        [
            "bash",
            "-lc",
            "./gradlew :env:installDist :tables:dataVersionTables "
            "-Pcrforge.assetSource=cdn -Pcrforge.dataVersion=16.635.3",
        ],
        cwd=vendor,
    )

    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "gymnasium>=1.0,<2",
            "pettingzoo>=1.24,<2",
            "jpype1>=1.5,<2",
            "PyYAML>=6",
            "msgpack>=1",
            "pyarrow>=18",
            "tqdm>=4.66",
        ]
    )
    run([sys.executable, "-m", "pip", "install", "--no-deps", "-e", str(ROOT)])
    run([sys.executable, "-m", "pip", "install", "--no-deps", "-e", str(vendor / "python")])

    env = os.environ.copy()
    env["JAVA_HOME"] = java_home
    env["PATH"] = os.environ["PATH"]
    print("=== Simulator doctor ===", flush=True)
    run([sys.executable, "-m", "cr_rl.cli", "doctor"], cwd=ROOT)
    print("=== General self-play PPO GPU smoke ===", flush=True)
    shutil.rmtree(ROOT / "artifacts/colab_selfplay_smoke", ignore_errors=True)
    run(
        [
            sys.executable,
            "scripts/train_selfplay.py",
            "--model",
            "general",
            "--envs",
            "2",
            "--steps",
            "32",
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--grad-accumulation",
            "2",
            "--device",
            "cuda",
            "--out",
            "artifacts/colab_selfplay_smoke",
        ],
        cwd=ROOT,
    )
    production_check = (
        "import torch; from cr_rl.models import ActorCritic, ModelConfig; "
        "c=ModelConfig.preset('production'); m=ActorCritic(c).cuda().eval(); "
        "h=torch.zeros(1,8,c.header_dim,device='cuda'); "
        "e=torch.zeros(1,8,c.max_entities,c.entity_feature_dim,device='cuda'); "
        "mask=torch.zeros(1,8,c.max_entities,dtype=torch.bool,device='cuda'); mask[:,:,0]=True; "
        "logits,value=m(h,e,mask); "
        "print('production_params',m.parameter_count(),'shapes',tuple(logits.shape), "
        "tuple(value.shape), "
        "'peak_gb',round(torch.cuda.max_memory_allocated()/2**30,3))"
    )
    run([sys.executable, "-c", production_check], cwd=ROOT)
    result_archive = Path("/content/colab_smoke_results.tar.gz")
    with tarfile.open(result_archive, "w:gz") as archive:
        archive.add(ROOT / "artifacts/colab_selfplay_smoke", arcname="colab_selfplay_smoke")
    print("=== Colab smoke complete ===", flush=True)


if __name__ == "__main__":
    main()
