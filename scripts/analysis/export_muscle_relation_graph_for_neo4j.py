from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_NORMAL_DATASET = PROJECT_ROOT / "artifacts" / "retrain_labeled_records" / "graph_sequence_dataset"
DEFAULT_ABNORMAL_DATASET = PROJECT_ROOT / "artifacts" / "retrain_labeled_records" / "abnormal_graph_sequence_dataset"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "final_book_figures" / "neo4j_export"


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none"} else text


def _muscle_group_from_edges(edges: list[dict[str, Any]], node_names: list[str]) -> dict[str, str]:
    votes: dict[str, Counter[str]] = {name: Counter() for name in node_names}
    for edge in edges:
        metadata = edge.get("metadata") or {}
        shared_groups = metadata.get("shared_groups") or []
        if not isinstance(shared_groups, list):
            continue
        for group in shared_groups:
            group_text = _safe_text(group)
            if not group_text:
                continue
            for key in ("source_name", "target_name"):
                node = _safe_text(edge.get(key))
                if node in votes:
                    votes[node][group_text] += 1
    groups: dict[str, str] = {}
    for node, counter in votes.items():
        if counter:
            groups[node] = counter.most_common(1)[0][0]
        elif node.startswith(("E", "LU", "UI", "RI")):
            groups[node] = "extensor_or_radial_ulnar_group"
        elif node.startswith(("F", "P")):
            groups[node] = "flexor_or_pronator_group"
        else:
            groups[node] = "muscle"
    return groups


def _manifest_paths(dataset_dir: Path) -> list[dict[str, str]]:
    manifest_path = dataset_dir / "dataset_manifest.csv"
    if not manifest_path.exists():
        return []
    manifest = pd.read_csv(manifest_path)
    rows: list[dict[str, str]] = []
    for item in manifest.to_dict(orient="records"):
        json_path = Path(_safe_text(item.get("json_path")))
        npz_path = Path(_safe_text(item.get("npz_path")))
        if json_path.exists() and npz_path.exists():
            rows.append({"json_path": str(json_path), "npz_path": str(npz_path), "cohort": _safe_text(item.get("cohort"))})
    return rows


def _load_graph_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def export_neo4j_relation_graph(*, normal_dataset: Path, abnormal_dataset: Path | None, output_dir: Path) -> dict[str, Any]:
    graph_rows = _manifest_paths(normal_dataset)
    if abnormal_dataset is not None:
        graph_rows.extend(_manifest_paths(abnormal_dataset))
    if not graph_rows:
        raise FileNotFoundError("No graph sequence manifest rows found for Neo4j export.")

    first_graph = _load_graph_json(Path(graph_rows[0]["json_path"]))
    node_names = [str(name) for name in first_graph["node_names"]]
    node_groups = _muscle_group_from_edges(list(first_graph.get("edges") or []), node_names)

    edge_accumulator: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    edge_prior: dict[tuple[str, str, str], float] = {}
    relation_counts: Counter[str] = Counter()

    for row in graph_rows:
        graph = _load_graph_json(Path(row["json_path"]))
        edges = list(graph.get("edges") or [])
        npz = np.load(Path(row["npz_path"]), allow_pickle=True)
        edge_weight_seq = np.asarray(npz["edge_weight_seq"], dtype=float)
        for edge in edges:
            edge_idx = int(edge.get("edge_idx", -1))
            if edge_idx < 0 or edge_idx >= edge_weight_seq.shape[1]:
                continue
            src = _safe_text(edge.get("source_name"))
            dst = _safe_text(edge.get("target_name"))
            relation_type = _safe_text(edge.get("edge_type")) or "unknown"
            if not src or not dst:
                continue
            key = (src, dst, relation_type)
            weight = float(np.nanmean(np.abs(edge_weight_seq[:, edge_idx])))
            if np.isfinite(weight):
                edge_accumulator[key].append(weight)
                edge_prior[key] = float(edge.get("prior_weight", 0.0) or 0.0)
                relation_counts[relation_type] += 1

    output_dir.mkdir(parents=True, exist_ok=True)
    nodes_path = output_dir / "neo4j_muscle_nodes.csv"
    edges_path = output_dir / "neo4j_muscle_edges.csv"
    readme_path = output_dir / "README.md"

    with nodes_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["muscle_id", "muscle_name", "muscle_group", "status"])
        writer.writeheader()
        for idx, name in enumerate(node_names):
            writer.writerow(
                {
                    "muscle_id": f"M{idx:03d}",
                    "muscle_name": name,
                    "muscle_group": node_groups.get(name, "muscle"),
                    "status": "reference_node",
                }
            )

    with edges_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["source_muscle", "target_muscle", "relation_type", "edge_weight", "aggregation_method"],
        )
        writer.writeheader()
        for (src, dst, relation_type), weights in sorted(edge_accumulator.items()):
            writer.writerow(
                {
                    "source_muscle": src,
                    "target_muscle": dst,
                    "relation_type": relation_type,
                    "edge_weight": f"{float(np.mean(weights)):.8f}",
                    "aggregation_method": "mean_abs_dynamic_edge_weight_over_progress_and_trials",
                }
            )

    readme_path.write_text(
        f"""# Neo4j Muscle Relation Graph Export

This folder contains CSV files for visualizing the muscle relation graph used by the Dynamic Temporal GCN workflow.

Neo4j is used only for graph visualization and exploratory queries. It is not part of model training.

## Files

- `neo4j_muscle_nodes.csv`: muscle/actuator nodes.
- `neo4j_muscle_edges.csv`: muscle relation edges with aggregated dynamic edge weight.

## Source

- Normal graph sequence dataset: `{normal_dataset}`
- Abnormal graph sequence dataset: `{abnormal_dataset if abnormal_dataset is not None else "not included"}`
- Edge aggregation: `mean_abs_dynamic_edge_weight_over_progress_and_trials`
- Node count: `{len(node_names)}`
- Edge count: `{len(edge_accumulator)}`

## Import With Neo4j Browser

Copy both CSV files into Neo4j's `import/` directory, then run:

```cypher
CREATE CONSTRAINT muscle_name_unique IF NOT EXISTS
FOR (m:Muscle) REQUIRE m.muscle_name IS UNIQUE;

LOAD CSV WITH HEADERS FROM 'file:///neo4j_muscle_nodes.csv' AS row
MERGE (m:Muscle {{muscle_name: row.muscle_name}})
SET m.muscle_id = row.muscle_id,
    m.muscle_group = row.muscle_group,
    m.status = row.status;

LOAD CSV WITH HEADERS FROM 'file:///neo4j_muscle_edges.csv' AS row
MATCH (src:Muscle {{muscle_name: row.source_muscle}})
MATCH (dst:Muscle {{muscle_name: row.target_muscle}})
MERGE (src)-[r:MUSCLE_RELATION {{
    relation_type: row.relation_type,
    aggregation_method: row.aggregation_method
}}]->(dst)
SET r.edge_weight = toFloat(row.edge_weight);
```

## Useful Queries

Show the full graph:

```cypher
MATCH p=(m:Muscle)-[r:MUSCLE_RELATION]->(n:Muscle)
RETURN p
LIMIT 300;
```

Show strongest relations:

```cypher
MATCH (m:Muscle)-[r:MUSCLE_RELATION]->(n:Muscle)
RETURN m.muscle_name AS source,
       n.muscle_name AS target,
       r.relation_type AS relation,
       r.edge_weight AS weight
ORDER BY weight DESC
LIMIT 50;
```

Show synergist/antagonist relations only:

```cypher
MATCH p=(m:Muscle)-[r:MUSCLE_RELATION]->(n:Muscle)
WHERE r.relation_type IN ['synergist', 'antagonist']
RETURN p
LIMIT 200;
```

Show degree per muscle:

```cypher
MATCH (m:Muscle)-[r:MUSCLE_RELATION]-()
RETURN m.muscle_name AS muscle,
       m.muscle_group AS muscle_group,
       count(r) AS relation_count,
       avg(r.edge_weight) AS mean_edge_weight
ORDER BY relation_count DESC;
```
""",
        encoding="utf-8",
    )

    summary = {
        "nodes_path": str(nodes_path),
        "edges_path": str(edges_path),
        "readme_path": str(readme_path),
        "node_count": len(node_names),
        "edge_count": len(edge_accumulator),
        "relation_counts": dict(relation_counts),
        "aggregation_method": "mean_abs_dynamic_edge_weight_over_progress_and_trials",
        "source_graph_rows": len(graph_rows),
    }
    (output_dir / "export_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Export Dynamic Temporal GCN muscle relation graph data for Neo4j visualization.")
    parser.add_argument("--normal-dataset", type=Path, default=DEFAULT_NORMAL_DATASET)
    parser.add_argument("--abnormal-dataset", type=Path, default=DEFAULT_ABNORMAL_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--normal-only", action="store_true", help="Export only normal graph sequence dataset.")
    args = parser.parse_args()

    summary = export_neo4j_relation_graph(
        normal_dataset=args.normal_dataset,
        abnormal_dataset=None if args.normal_only else args.abnormal_dataset,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
