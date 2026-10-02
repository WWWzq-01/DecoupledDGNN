#!/usr/bin/env python3
"""Verify the DecoupledDGNN runtime against benchmark-local data."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
if str(SYSTEM_ROOT) not in sys.path:
    sys.path.insert(0, str(SYSTEM_ROOT))


REQUIRED_MODULES = (
    "torch",
    "torch_geometric",
    "numpy",
    "pandas",
    "sklearn",
    "ogb",
    "scipy",
    "tqdm",
    "Cython",
    "eigency",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="/home/phc/data")
    parser.add_argument("--dataset", default="lastfm")
    parser.add_argument("--edge-count", type=int, default=8)
    parser.add_argument("--feature-dim", type=int, default=8)
    return parser.parse_args()


def import_required_modules() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in REQUIRED_MODULES:
        module = importlib.import_module(name)
        versions[name] = str(getattr(module, "__version__", "ok"))
    return versions


def load_edges(dataset_dir: Path, edge_count: int) -> pd.DataFrame:
    for filename in ("edges.csv", "train.csv"):
        path = dataset_dir / filename
        if path.exists():
            frame = pd.read_csv(path, nrows=max(edge_count * 4, edge_count))
            if "src" not in frame.columns or "dst" not in frame.columns:
                raise ValueError(f"{path} must contain src and dst columns")
            frame = frame[["src", "dst"]].dropna().head(edge_count)
            if len(frame) < edge_count:
                raise ValueError(f"{path} has only {len(frame)} valid edges")
            frame = frame.astype({"src": "int64", "dst": "int64"})
            return frame
    raise FileNotFoundError(f"No edges.csv or train.csv found in {dataset_dir}")


def load_node_features(dataset_dir: Path) -> np.ndarray:
    path = dataset_dir / "nfeats.npy"
    if not path.exists():
        raise FileNotFoundError(f"Missing node feature file: {path}")
    return np.load(path, mmap_mode="r")


def build_smoke_graph(
    edges: pd.DataFrame,
    node_features: np.ndarray,
    edge_count: int,
    feature_dim: int,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    node_ids = sorted(set(edges["src"].tolist()) | set(edges["dst"].tolist()))
    if max(node_ids) >= node_features.shape[0]:
        raise ValueError(
            f"Edge node id {max(node_ids)} exceeds nfeats rows {node_features.shape[0]}"
        )
    remap = {node_id: idx for idx, node_id in enumerate(node_ids)}
    remapped_edges = [
        (remap[int(row.src)], remap[int(row.dst)])
        for row in edges.itertuples(index=False)
    ][:edge_count]
    dim = min(feature_dim, node_features.shape[1])
    features = np.asarray(node_features[node_ids, :dim], dtype=np.float64)
    return np.ascontiguousarray(features), remapped_edges


def run_instantgnn_smoke(
    features: np.ndarray,
    remapped_edges: list[tuple[int, int]],
) -> float:
    from propagation import InstantGNN

    with tempfile.TemporaryDirectory(prefix="decoupleddgnn-smoke-") as tmpdir:
        tmp_path = Path(tmpdir)
        dataset_name = "benchmark_smoke"
        edge_file = tmp_path / f"{dataset_name}.txt"
        edge_file.write_text(
            "\n".join(f"{src} {dst}" for src, dst in remapped_edges) + "\n",
            encoding="utf-8",
        )
        graph = InstantGNN()
        return float(
            graph.initial_operation(
                str(tmp_path) + "/",
                dataset_name,
                len(remapped_edges),
                features.shape[0],
                1e-7,
                0.2,
                features,
            )
        )


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.data_root) / args.dataset
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")

    versions = import_required_modules()
    propagation = importlib.import_module("propagation")
    edges = load_edges(dataset_dir, args.edge_count)
    node_features = load_node_features(dataset_dir)
    features, remapped_edges = build_smoke_graph(
        edges,
        node_features,
        args.edge_count,
        args.feature_dim,
    )
    dataset_size_gib = run_instantgnn_smoke(features, remapped_edges)

    print(
        json.dumps(
            {
                "status": "ok",
                "dataset": args.dataset,
                "dataset_dir": str(dataset_dir),
                "propagation_module": str(Path(propagation.__file__).resolve()),
                "python_versions": versions,
                "source_node_feature_shape": list(node_features.shape),
                "smoke_node_count": int(features.shape[0]),
                "smoke_feature_dim": int(features.shape[1]),
                "smoke_edge_count": len(remapped_edges),
                "instantgnn_dataset_size_gib": dataset_size_gib,
            },
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
