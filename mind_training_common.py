import collections
import os
from pathlib import Path
from urllib.request import urlretrieve
import zipfile


REPO_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = REPO_ROOT / "dataset"
WIKIDATA_GRAPH_URL = "https://mind201910.blob.core.windows.net/knowledge-graph/wikidata-graph.zip"


def get_mind_dataset_paths(size: str):
    if size not in {"small", "large"}:
        raise ValueError(f"Unsupported MIND size: {size}")
    prefix = "MINDsmall" if size == "small" else "MINDlarge"
    return {
        "train": DATASET_ROOT / f"{prefix}_train",
        "dev": DATASET_ROOT / f"{prefix}_dev",
        "test": DATASET_ROOT / f"{prefix}_test",
    }


def build_codabench_submission_artifacts(ranking_file, output_dir):
    ranking_path = Path(ranking_file)
    submission_dir = Path(output_dir)
    submission_dir.mkdir(parents=True, exist_ok=True)

    prediction_txt = submission_dir / "prediction.txt"
    content = ranking_path.read_text(encoding="utf-8").replace("\r\n", "\n")
    with open(prediction_txt, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(content)

    prediction_zip = submission_dir / "prediction.zip"
    with zipfile.ZipFile(prediction_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(prediction_txt, arcname="prediction.txt")

    return prediction_txt, prediction_zip


def ensure_mind_context_embeddings(size: str):
    split_paths = get_mind_dataset_paths(size)
    missing_splits = [name for name, path in split_paths.items() if not (path / "context_embedding.vec").exists()]
    if not missing_splits:
        return

    graph_path = ensure_wikidata_graph()
    relations = load_wikidata_relations(graph_path)
    entity_embeddings = load_entity_embeddings(split_paths.values())
    context_embeddings = build_context_embeddings(entity_embeddings, relations)

    for split_name in missing_splits:
        write_context_embeddings(
            entity_embedding_file=split_paths[split_name] / "entity_embedding.vec",
            context_embedding_file=split_paths[split_name] / "context_embedding.vec",
            context_embeddings=context_embeddings,
        )


def ensure_wikidata_graph():
    download_root = DATASET_ROOT / "download"
    zip_path = download_root / "wikidata-graph.zip"
    extract_root = download_root / "wikidata-graph"
    graph_path = extract_root / "wikidata-graph.tsv"

    if graph_path.exists():
        return graph_path

    download_root.mkdir(parents=True, exist_ok=True)
    if not zip_path.exists():
        urlretrieve(WIKIDATA_GRAPH_URL, zip_path)

    extract_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(download_root)

    if not graph_path.exists():
        raise FileNotFoundError(f"wikidata graph not found after extraction: {graph_path}")
    return graph_path


def load_wikidata_relations(graph_path):
    relations = collections.defaultdict(set)
    with open(graph_path, "r", encoding="utf-8") as graph_file:
        for line in graph_file:
            if not line.strip():
                continue
            head, _, tail = line.rstrip("\n").split("\t")
            relations[head].add(tail)
            relations[tail].add(head)
    return relations


def load_entity_embeddings(split_dirs):
    entity_embeddings = {}
    for split_dir in split_dirs:
        entity_file = Path(split_dir) / "entity_embedding.vec"
        with open(entity_file, "r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                parts = line.rstrip("\n").split("\t")
                entity_embeddings[parts[0]] = [float(value) for value in parts[1:]]
    return entity_embeddings


def build_context_embeddings(entity_embeddings, relations):
    context_embeddings = {}
    for entity, base_embedding in entity_embeddings.items():
        context = list(base_embedding)
        count = 1
        for neighbor in relations.get(entity, ()):
            if neighbor in entity_embeddings:
                neighbor_embedding = entity_embeddings[neighbor]
                for index, value in enumerate(neighbor_embedding):
                    context[index] += value
                count += 1
        context_embeddings[entity] = [value / count for value in context]
    return context_embeddings


def write_context_embeddings(entity_embedding_file, context_embedding_file, context_embeddings):
    with open(entity_embedding_file, "r", encoding="utf-8") as entity_file, open(
        context_embedding_file, "w", encoding="utf-8"
    ) as context_file:
        for line in entity_file:
            if not line.strip():
                continue
            entity = line.split("\t", 1)[0]
            values = context_embeddings[entity]
            context_file.write(entity + "\t" + "\t".join(map(str, values)) + "\n")


def write_results_summary(summary_path, payload):
    summary_file = Path(summary_path)
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{key}={value}" for key, value in payload.items()]
    summary_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
