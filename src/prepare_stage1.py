"""Resolve or build reusable Stage-1 data and jet-attribute-model artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


CACHE_VERSION = 1
JET_TYPE_TO_INDEX = {name: i for i, name in enumerate(("g", "q", "t", "w", "z"))}
MODEL_SPEC = {
    "random_seed": 42,
    "num_epochs": 35_000,
    "batch_size": 8192,
    "K": 10,
    "hidden_units": 128,
    "hidden_layers": 8,
}


def stage1_spec(jet_types: list[str], num_particles: int) -> dict:
    unknown = sorted(set(jet_types) - JET_TYPE_TO_INDEX.keys())
    if unknown:
        raise ValueError(f"unknown jet types: {unknown}")
    return {
        "cache_version": CACHE_VERSION,
        "jet_types": sorted(set(jet_types), key=JET_TYPE_TO_INDEX.__getitem__),
        "num_particles": int(num_particles),
        "model": MODEL_SPEC,
    }


def cache_key(spec: dict) -> str:
    payload = json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(payload).hexdigest()[:12]
    types = "-".join(spec["jet_types"])
    return f"{types}-p{spec['num_particles']}-v{CACHE_VERSION}-{digest}"


def validate_bundle(bundle: Path, spec: dict, require_metadata: bool = True) -> None:
    required = [bundle / "data" / "x_train.pkl", bundle / "data" / "x_test.pkl",
                bundle / "jet_attr_model.pth"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"incomplete Stage-1 bundle; missing {missing}")

    metadata_path = bundle / "stage1_metadata.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("spec") != spec:
            raise ValueError(f"Stage-1 metadata mismatch in {bundle}")
        return
    if require_metadata:
        raise FileNotFoundError(f"missing Stage-1 metadata: {metadata_path}")

    # Legacy training runs predate cache metadata but record the data contract in
    # their compact summary. Prefer it over unpickling a large JetNet object from
    # network storage; retain the pickle path for older runs without summaries.
    summary_path = bundle / "train" / "summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text())
        data_config = summary.get("full_config", {}).get("data", {})
        actual_spec = stage1_spec(
            data_config.get("jet_types", []), data_config.get("num_particles", -1)
        )
        if actual_spec["jet_types"] != spec["jet_types"]:
            raise ValueError(
                f"legacy Stage-1 types {actual_spec['jet_types']} != {spec['jet_types']}"
            )
        if actual_spec["num_particles"] != spec["num_particles"]:
            raise ValueError(
                f"legacy Stage-1 particle count {actual_spec['num_particles']} "
                f"!= {spec['num_particles']}"
            )
        return

    # Oldest runs have no summary; validate their actual data as a fallback.
    with (bundle / "data" / "x_train.pkl").open("rb") as handle:
        particles, jet_features = pickle.load(handle)
    if particles.shape[1] != spec["num_particles"]:
        raise ValueError(
            f"legacy Stage-1 particle count {particles.shape[1]} != {spec['num_particles']}"
        )
    actual_types = sorted({int(v) for v in jet_features[:, -1].long().tolist()})
    expected_types = sorted(JET_TYPE_TO_INDEX[name] for name in spec["jet_types"])
    if actual_types != expected_types:
        raise ValueError(f"legacy Stage-1 types {actual_types} != {expected_types}")


def publish_bundle(source: Path, destination: Path, spec: dict, source_label: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        validate_bundle(destination, spec)
        return
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        shutil.copytree(source / "data", staging / "data")
        shutil.copy2(source / "jet_attr_model.pth", staging / "jet_attr_model.pth")
        (staging / "stage1_metadata.json").write_text(json.dumps({
            "spec": spec,
            "source": source_label,
        }, indent=2) + "\n")
        try:
            staging.rename(destination)
        except FileExistsError:
            validate_bundle(destination, spec)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def link_bundle(bundle: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for name in ("data", "jet_attr_model.pth"):
        target = output / name
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"refusing to replace existing Stage-1 output: {target}")
        target.symlink_to(bundle / name, target_is_directory=(name == "data"))
    (output / "stage1_source.txt").write_text(str(bundle.resolve()) + "\n")


def resolve_stage1(
    output: Path,
    cache_root: Path,
    jet_types: list[str],
    num_particles: int,
    import_run: Path | None = None,
) -> tuple[Path, bool]:
    spec = stage1_spec(jet_types, num_particles)
    bundle = cache_root / cache_key(spec)
    if bundle.exists():
        validate_bundle(bundle, spec)
        link_bundle(bundle, output)
        return bundle, True

    if import_run is not None:
        validate_bundle(import_run, spec, require_metadata=False)
        publish_bundle(import_run, bundle, spec, str(import_run.resolve()))
        link_bundle(bundle, output)
        return bundle, True

    repo_src = Path(__file__).resolve().parent
    output.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        sys.executable, str(repo_src / "data.py"), "--jet_types", *spec["jet_types"],
        "--num_particles", str(num_particles), "--output_path", str(output),
    ], check=True, cwd=repo_src)
    subprocess.run([
        sys.executable, str(repo_src / "jet_attr_model.py"), "--output_path", str(output),
    ], check=True, cwd=repo_src)
    validate_bundle(output, spec, require_metadata=False)
    publish_bundle(output, bundle, spec, "trained")
    (output / "stage1_source.txt").write_text(str(bundle.resolve()) + "\n")
    return bundle, False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("/mnt/data/caches/stage1"))
    parser.add_argument("--jet-types", nargs="+", required=True)
    parser.add_argument("--num-particles", type=int, required=True)
    parser.add_argument("--import-run", type=Path)
    args = parser.parse_args()
    bundle, reused = resolve_stage1(
        args.output_path, args.cache_dir, args.jet_types, args.num_particles, args.import_run
    )
    print(f"Stage-1 {'cache hit' if reused else 'trained and cached'}: {bundle}", flush=True)


if __name__ == "__main__":
    main()
