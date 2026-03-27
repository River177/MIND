# train_nrms_large_submit.py

import os
import pickle
import zipfile
import tempfile
import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from recommenders.models.newsrec.io.mind_iterator import MINDIterator
from recommenders.models.newsrec.models.nrms import NRMSModel
from recommenders.models.newsrec.newsrec_utils import prepare_hparams


def get_default_base_dir():
    return os.environ.get("MIND_BASE_DIR", str(Path(__file__).resolve().parent))


BASE_DIR = get_default_base_dir()
DATA_DIR = os.path.join(BASE_DIR, "dataset")
UTILS_DIR = os.path.join(DATA_DIR, "utils")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# ===== 改成 large =====
TRAIN_DIR = os.path.join(DATA_DIR, "MINDlarge_train")
DEV_DIR = os.path.join(DATA_DIR, "MINDlarge_dev")
TEST_DIR = os.path.join(DATA_DIR, "MINDlarge_test")

train_news_file = os.path.join(TRAIN_DIR, "news.tsv")
train_behaviors_file = os.path.join(TRAIN_DIR, "behaviors.tsv")

valid_news_file = os.path.join(DEV_DIR, "news.tsv")
valid_behaviors_file = os.path.join(DEV_DIR, "behaviors.tsv")

test_news_file = os.path.join(TEST_DIR, "news.tsv")
test_behaviors_file = os.path.join(TEST_DIR, "behaviors.tsv")

# ===== 这里默认你已经用 LARGE 重新生成过 utils =====
wordEmb_file = os.path.join(UTILS_DIR, "embedding.npy")
wordDict_file = os.path.join(UTILS_DIR, "word_dict.pkl")
userDict_file = os.path.join(UTILS_DIR, "uid2index.pkl")


def check_exists(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"文件不存在: {path}")


def show_basic_info():
    with open(wordDict_file, "rb") as f:
        word_dict = pickle.load(f)
    with open(userDict_file, "rb") as f:
        user_dict = pickle.load(f)

    print("word_dict size:", len(word_dict))
    print("user_dict size:", len(user_dict))


def setup_gpu():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"检测到 {len(gpus)} 张 GPU，已开启 memory growth")
        except Exception as e:
            print("GPU 设置失败：", e)
    else:
        print("未检测到 GPU，将使用 CPU")


def build_hparams():
    hparams = prepare_hparams(
        yaml_file=None,

        # 必需
        model_type="nrms",
        data_format="news",
        wordEmb_file=wordEmb_file,
        wordDict_file=wordDict_file,
        userDict_file=userDict_file,

        # 数据相关
        title_size=30,
        his_size=50,
        npratio=4,
        word_emb_dim=300,   # 你现在是 GloVe 840B 300d

        # NRMS 结构
        head_num=20,
        head_dim=20,
        attention_hidden_dim=200,
        dropout=0.2,

        # 训练
        learning_rate=1e-4,
        optimizer="adam",
        loss="cross_entropy_loss",
        epochs=3,
        batch_size=32,
        show_step=100,

        # 评估
        support_quick_scoring=True,
        metrics=["group_auc", "mean_mrr", "ndcg@5;10"],
    )
    return hparams


def parse_test_behaviors(behaviors_path):
    """
    解析 MINDlarge_test/behaviors.tsv
    返回:
        impression_ids: [str, ...]
        candidate_news_list: [[nid1, nid2, ...], ...]
    兼容两种格式:
        1) test:   N1 N2 N3 ...
        2) dev/train: N1-0 N2-1 ...
    """
    impression_ids = []
    candidate_news_list = []

    with open(behaviors_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue

            parts = line.split("\t")
            if len(parts) < 5:
                continue

            impression_id = parts[0]
            impr = parts[4].strip()

            if impr == "":
                candidates = []
            else:
                raw_items = impr.split()
                candidates = []
                for item in raw_items:
                    # dev/train 常见: N12345-0
                    # test 常见: N12345
                    if "-" in item:
                        nid = item.rsplit("-", 1)[0]
                    else:
                        nid = item
                    candidates.append(nid)

            impression_ids.append(impression_id)
            candidate_news_list.append(candidates)

    return impression_ids, candidate_news_list


def parse_submission_line(line):
    impression_id, ranks = line.strip("\n").split()
    return impression_id, list(map(int, ranks.strip("[]").split(","))) if ranks.strip("[]") else []


def create_truth_file_from_behaviors(behaviors_file, truth_file):
    with open(behaviors_file, "r", encoding="utf-8") as src, open(truth_file, "w", encoding="utf-8") as dst:
        for line in src:
            line = line.rstrip("\n")
            if not line:
                continue

            parts = line.split("\t")
            if len(parts) < 5:
                continue

            impression_id = parts[0]
            impressions = parts[4].strip()
            labels = []
            if impressions:
                for item in impressions.split():
                    labels.append(int(item.rsplit("-", 1)[1]) if "-" in item else 0)
            dst.write(f"{impression_id} [{','.join(map(str, labels))}]\n")


def roc_auc_from_ranks(labels, ranks):
    positives = [rank for label, rank in zip(labels, ranks) if label == 1]
    negatives = [rank for label, rank in zip(labels, ranks) if label == 0]
    if not positives or not negatives:
        return 0.5

    wins = 0.0
    total = len(positives) * len(negatives)
    for p_rank in positives:
        p_score = 1.0 / p_rank
        for n_rank in negatives:
            n_score = 1.0 / n_rank
            if p_score > n_score:
                wins += 1.0
            elif p_score == n_score:
                wins += 0.5
    return wins / total


def mrr_from_ranks(labels, ranks):
    ordered = [label for _, label in sorted(zip(ranks, labels), key=lambda x: x[0])]
    positive_count = sum(ordered)
    if positive_count == 0:
        return 0.0
    rr_score = sum(label / (idx + 1) for idx, label in enumerate(ordered))
    return rr_score / positive_count


def dcg_from_ranks(labels, ranks, k):
    ordered = [label for _, label in sorted(zip(ranks, labels), key=lambda x: x[0])][:k]
    return sum((2 ** label - 1) / np.log2(idx + 2) for idx, label in enumerate(ordered))


def ndcg_from_ranks(labels, ranks, k):
    best = dcg_from_ranks(labels, list(range(1, len(labels) + 1)), k) if any(labels) else 0.0
    if best == 0.0:
        return 0.0
    ideal_labels = sorted(labels, reverse=True)
    best = sum((2 ** label - 1) / np.log2(idx + 2) for idx, label in enumerate(ideal_labels[:k]))
    actual = dcg_from_ranks(labels, ranks, k)
    return actual / best


def score_submission_file(truth_file, prediction_file):
    aucs = []
    mrrs = []
    ndcg5s = []
    ndcg10s = []

    with open(truth_file, "r", encoding="utf-8") as truth_f, open(prediction_file, "r", encoding="utf-8") as pred_f:
        for line_idx, truth_line in enumerate(truth_f, start=1):
            pred_line = pred_f.readline()
            truth_impression_id, labels = parse_submission_line(truth_line)
            if labels == []:
                continue

            if pred_line == "":
                pred_impression_id, ranks = truth_impression_id, [1] * len(labels)
            else:
                pred_impression_id, ranks = parse_submission_line(pred_line)

            if pred_impression_id != truth_impression_id:
                raise ValueError(
                    f"line-{line_idx}: Inconsistent Impression Id {pred_impression_id} and {truth_impression_id}"
                )
            if len(ranks) != len(labels):
                raise ValueError(f"line-{line_idx}: Rank count {len(ranks)} does not match label count {len(labels)}")

            aucs.append(roc_auc_from_ranks(labels, ranks))
            mrrs.append(mrr_from_ranks(labels, ranks))
            ndcg5s.append(ndcg_from_ranks(labels, ranks, 5))
            ndcg10s.append(ndcg_from_ranks(labels, ranks, 10))

    return {
        "group_auc": round(float(np.mean(aucs)), 4) if aucs else 0.0,
        "mean_mrr": round(float(np.mean(mrrs)), 4) if mrrs else 0.0,
        "ndcg@5": round(float(np.mean(ndcg5s)), 4) if ndcg5s else 0.0,
        "ndcg@10": round(float(np.mean(ndcg10s)), 4) if ndcg10s else 0.0,
    }


def ranks_from_scores(scores):
    """
    输入一组分数，输出排名列表（1=最高名次）
    例如 scores=[0.2, 0.8, 0.5] -> [3,1,2]
    """
    order = np.argsort(scores)[::-1]  # 从高到低的索引
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(scores) + 1)
    return ranks.tolist()


def write_prediction_file(impression_ids, group_preds, out_txt):
    """
    写出 Codabench / MIND 提交文件:
    每行: impression_id [rank1,rank2,...]
    """
    if len(impression_ids) != len(group_preds):
        raise ValueError(
            f"impression 数量和预测组数不一致: {len(impression_ids)} vs {len(group_preds)}"
        )

    with open(out_txt, "w", encoding="utf-8") as f:
        for impr_id, preds in tqdm(
            zip(impression_ids, group_preds),
            total=len(impression_ids),
            desc="写 prediction.txt"
        ):
            preds = np.asarray(preds, dtype=np.float32)
            pred_rank = ranks_from_scores(preds)
            rank_str = "[" + ",".join(map(str, pred_rank)) + "]"
            f.write(f"{impr_id} {rank_str}\n")


def zip_prediction_file(txt_path, zip_path):
    """
    zip 内只能放 prediction.txt
    """
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(txt_path, arcname="prediction.txt")


def lookup_user_vector(user_vecs, impression_id, fallback_idx=None):
    candidate_keys = []
    if fallback_idx is not None:
        candidate_keys.append(fallback_idx)
    candidate_keys.append(impression_id)
    try:
        candidate_keys.append(int(impression_id))
    except (TypeError, ValueError):
        pass

    for key in candidate_keys:
        if key in user_vecs:
            return user_vecs[key]

    sample_keys = list(user_vecs.keys())[:5]
    raise KeyError(
        f"Cannot find user vector for impression_id={impression_id}, "
        f"fallback_idx={fallback_idx}. Sample keys: {sample_keys}"
    )


def lookup_news_vector(news_vecs, nid2index, news_id):
    news_index = nid2index.get(news_id)
    if news_index is None:
        return None
    return news_vecs.get(news_index)


def create_run_user_behaviors_file(behaviors_file):
    temp_file = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".tsv",
        prefix="nrms_run_user_",
        delete=False,
    )
    try:
        with open(behaviors_file, "r", encoding="utf-8") as src:
            for line in src:
                line = line.rstrip("\n")
                if not line:
                    temp_file.write("\n")
                    continue

                parts = line.split("\t")
                if len(parts) < 5:
                    temp_file.write(line + "\n")
                    continue

                impressions = parts[4].strip()
                if impressions:
                    normalized = []
                    for item in impressions.split():
                        normalized.append(item if "-" in item else f"{item}-0")
                    parts[4] = " ".join(normalized)
                temp_file.write("\t".join(parts) + "\n")
    finally:
        temp_file.close()

    return temp_file.name


def reset_test_iterator_state(model):
    iterator = getattr(model, "test_iterator", None)
    if iterator is None:
        return
    for attr in [
        "nid2index",
        "news_title_index",
        "histories",
        "imprs",
        "labels",
        "impr_indexes",
        "uindexes",
    ]:
        if hasattr(iterator, attr):
            delattr(iterator, attr)


def generate_test_submission(model, news_file, behaviors_file, out_dir, output_basename="prediction"):
    """
    为 MINDlarge_test 生成 prediction.txt / prediction.zip

    思路：
    1. 先用模型预计算 news embedding
    2. 再预计算每个 impression 的 user embedding
    3. 自己解析 test behaviors，拿到 impression_id 和候选新闻列表
    4. 点积打分 -> 转排名 -> 写 prediction.txt -> 压缩 zip

    这样不依赖 test 是否带 label。
    """
    print("开始为 test 集生成向量...")
    reset_test_iterator_state(model)
    news_vecs = model.run_news(news_file)
    nid2index = getattr(model.test_iterator, "nid2index", {})
    run_user_behaviors_file = create_run_user_behaviors_file(behaviors_file)
    try:
        user_vecs = model.run_user(news_file, run_user_behaviors_file)
    finally:
        if os.path.exists(run_user_behaviors_file):
            os.remove(run_user_behaviors_file)

    print("解析 test behaviors...")
    impression_ids, candidate_news_list = parse_test_behaviors(behaviors_file)

    print("开始计算 test 分数...")
    group_preds = []

    for idx, (impression_id, candidates) in tqdm(
        enumerate(zip(impression_ids, candidate_news_list)),
        total=len(candidate_news_list),
        desc="计算 test 打分"
    ):
        user_vec = lookup_user_vector(user_vecs, impression_id=impression_id, fallback_idx=idx)

        cand_vecs = []
        for nid in candidates:
            news_vec = lookup_news_vector(news_vecs, nid2index, nid)
            if news_vec is not None:
                cand_vecs.append(news_vec)
            else:
                # 理论上很少发生；如果发生，补零向量
                cand_vecs.append(np.zeros_like(user_vec, dtype=np.float32))

        if len(cand_vecs) == 0:
            preds = np.array([], dtype=np.float32)
        else:
            cand_vecs = np.stack(cand_vecs, axis=0)   # [num_candidates, dim]
            preds = np.dot(cand_vecs, user_vec)       # [num_candidates]

        group_preds.append(preds)

    txt_path = os.path.join(out_dir, f"{output_basename}.txt")
    zip_path = os.path.join(out_dir, f"{output_basename}.zip")

    print("写出 prediction.txt ...")
    write_prediction_file(impression_ids, group_preds, txt_path)

    print("压缩 prediction.zip ...")
    zip_prediction_file(txt_path, zip_path)

    print("提交文件已生成：")
    print("  ", txt_path)
    print("  ", zip_path)
    return txt_path, zip_path


def build_model(load_weights_path=None):
    hparams = build_hparams()
    model = NRMSModel(hparams, MINDIterator, seed=42)
    if load_weights_path:
        model.model.load_weights(load_weights_path)
        print("加载权重:", load_weights_path)
    return model


def validate_dev_submission(model, out_dir):
    prediction_txt, prediction_zip = generate_test_submission(
        model=model,
        news_file=valid_news_file,
        behaviors_file=valid_behaviors_file,
        out_dir=out_dir,
        output_basename="dev_prediction",
    )
    truth_path = os.path.join(out_dir, "dev_truth.txt")
    create_truth_file_from_behaviors(valid_behaviors_file, truth_path)
    metrics = score_submission_file(truth_path, prediction_txt)
    print("dev submission metrics:", metrics)
    return {
        "prediction_txt": prediction_txt,
        "prediction_zip": prediction_zip,
        "truth_txt": truth_path,
        "metrics": metrics,
    }


def main():
    parser = argparse.ArgumentParser(description="NRMS train / submission utility")
    parser.add_argument(
        "--operation",
        type=str,
        default="train",
        choices=["train", "submit_dev", "submit_test"],
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=os.path.join(OUTPUT_DIR, "nrms_large_weights.h5"),
    )
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. 检查文件
    for path in [
        train_news_file,
        train_behaviors_file,
        valid_news_file,
        valid_behaviors_file,
        test_news_file,
        test_behaviors_file,
        wordEmb_file,
        wordDict_file,
        userDict_file,
    ]:
        check_exists(path)

    # 2. GPU
    setup_gpu()

    # 3. 打印 utils 信息
    show_basic_info()

    if args.operation != "train":
        check_exists(args.weights)
        print("寮€濮嬫瀯寤?NRMS 妯″瀷...")
        model = build_model(load_weights_path=args.weights)
        if args.operation == "submit_dev":
            validate_dev_submission(model, OUTPUT_DIR)
        else:
            print("寮€濮嬬敓鎴?Codabench 鎻愪氦鏂囦欢锛圡INDlarge_test锛?..")
            generate_test_submission(
                model=model,
                news_file=test_news_file,
                behaviors_file=test_behaviors_file,
                out_dir=OUTPUT_DIR,
            )
        return

    # 4. 超参数
    hparams = build_hparams()

    # 5. 构建模型
    print("开始构建 NRMS 模型...")
    model = NRMSModel(hparams, MINDIterator, seed=42)

    # 6. 训练
    print("开始训练（MINDlarge_train）...")
    model.fit(
        train_news_file,
        train_behaviors_file,
        valid_news_file,
        valid_behaviors_file,
    )

    # 7. dev 评估
    print("开始 dev 集评估（MINDlarge_dev）...")
    res = model.run_eval(valid_news_file, valid_behaviors_file)
    print("dev 结果:", res)

    # 8. 保存权重
    weight_path = os.path.join(OUTPUT_DIR, "nrms_large_weights.h5")
    model.model.save_weights(weight_path)
    print("模型权重已保存到:", weight_path)

    # 9. 生成 Codabench 提交文件
    print("开始生成 Codabench 提交文件（MINDlarge_test）...")
    generate_test_submission(
        model=model,
        news_file=test_news_file,
        behaviors_file=test_behaviors_file,
        out_dir=OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()
