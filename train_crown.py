import importlib
import os
from pathlib import Path
import re
import sys

import torch

from mind_training_common import (
    build_codabench_submission_artifacts,
    ensure_mind_context_embeddings,
    write_results_summary,
)


ROOT = Path(__file__).resolve().parent
PROJECT_DIR = ROOT / "crown-www25"
DEFAULT_ARGS = [
    "--dataset",
    "large",
    "--mode",
    "train",
    "--news_encoder",
    "CROWN",
    "--user_encoder",
    "CROWN",
]


def import_project_modules():
    if str(PROJECT_DIR) not in sys.path:
        sys.path.insert(0, str(PROJECT_DIR))
    config_module = importlib.import_module("config")
    corpus_module = importlib.import_module("corpus")
    model_module = importlib.import_module("model")
    main_module = importlib.import_module("main")
    util_module = importlib.import_module("util")
    return config_module, corpus_module, model_module, main_module, util_module


def build_config(config_cls, cli_args):
    original_argv = sys.argv[:]
    try:
        sys.argv = [str(PROJECT_DIR / "main.py"), *DEFAULT_ARGS, *cli_args]
        return config_cls()
    finally:
        sys.argv = original_argv


def extract_run_index(path_text):
    match = re.search(r"#(\d+)", path_text)
    return int(match.group(1)) if match else 0


def evaluate_test(config, corpus, model_cls, compute_scores):
    model = model_cls(config)
    if not os.path.exists(config.test_model_path):
        raise FileNotFoundError(f"Test model does not exist: {config.test_model_path}")

    state = torch.load(config.test_model_path, map_location=torch.device("cpu"))
    model.load_state_dict(state[model.model_name])
    model.cuda()

    test_res_dir = os.path.join(
        config.test_res_dir,
        config.test_model_path.replace("\\", "_").replace("/", "_"),
    )
    os.makedirs(test_res_dir, exist_ok=True)
    ranking_file = os.path.join(test_res_dir, f"{model.model_name}.txt")
    auc, mrr, ndcg5, ndcg10 = compute_scores(
        model,
        corpus,
        config.batch_size,
        "test",
        ranking_file,
        config.dataset,
    )
    return ranking_file, auc, mrr, ndcg5, ndcg10


def main():
    os.chdir(ROOT)
    ensure_mind_context_embeddings("large")

    config_module, corpus_module, model_module, main_module, util_module = import_project_modules()
    config = build_config(config_module.Config, sys.argv[1:])
    corpus = corpus_module.Corpus(config)

    if config.mode == "train":
        main_module.train(config, corpus)
        model_name = f"{config.news_encoder}-{config.user_encoder}"
        config.test_model_path = os.path.join(config.best_model_dir, f"#{config.run_index}", model_name)
    elif config.mode == "test":
        config.run_index = extract_run_index(config.test_model_path)
    else:
        raise ValueError("train_crown.py supports train and test modes only")

    ranking_file, auc, mrr, ndcg5, ndcg10 = evaluate_test(
        config,
        corpus,
        model_module.Model,
        util_module.compute_scores,
    )

    submission_dir = Path(config.prediction_dir) / f"#{config.run_index}"
    prediction_txt, prediction_zip = build_codabench_submission_artifacts(ranking_file, submission_dir)

    summary_path = Path(config.result_dir) / f"#{config.run_index}-summary.txt"
    write_results_summary(
        summary_path,
        {
            "run_index": config.run_index,
            "dataset": config.dataset,
            "best_model_path": config.test_model_path,
            "dev_metrics_file": Path(config.result_dir) / f"#{config.run_index}-dev",
            "test_ranking_file": ranking_file,
            "prediction_txt": prediction_txt,
            "prediction_zip": prediction_zip,
            "auc": auc,
            "mrr": mrr,
            "ndcg5": ndcg5,
            "ndcg10": ndcg10,
        },
    )

    print(f"ranking_file={ranking_file}")
    print(f"prediction_txt={prediction_txt}")
    print(f"prediction_zip={prediction_zip}")
    if auc is not None:
        print(f"auc={auc:.4f}")
        print(f"mrr={mrr:.4f}")
        print(f"ndcg5={ndcg5:.4f}")
        print(f"ndcg10={ndcg10:.4f}")


if __name__ == "__main__":
    main()
