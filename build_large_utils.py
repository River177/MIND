import argparse
import os
import re
import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm


def get_default_base_dir():
    return os.environ.get("MIND_BASE_DIR", str(Path(__file__).resolve().parent))


BASE_DIR = get_default_base_dir()
DATA_DIR = os.path.join(BASE_DIR, "dataset")

TRAIN_DIR = os.path.join(DATA_DIR, "MINDlarge_train")
DEV_DIR = os.path.join(DATA_DIR, "MINDlarge_dev")
UTILS_DIR = os.path.join(DATA_DIR, "utils")
GLOVE_DIR = os.path.join(DATA_DIR, "glove")

TRAIN_NEWS = os.path.join(TRAIN_DIR, "news.tsv")
DEV_NEWS = os.path.join(DEV_DIR, "news.tsv")
TRAIN_BEHAVIORS = os.path.join(TRAIN_DIR, "behaviors.tsv")
DEV_BEHAVIORS = os.path.join(DEV_DIR, "behaviors.tsv")

DEFAULT_GLOVE_FILE = os.path.join(GLOVE_DIR, "glove.840B.300d.txt")

WORD_DICT_FILE = os.path.join(UTILS_DIR, "word_dict.pkl")
USER_DICT_FILE = os.path.join(UTILS_DIR, "uid2index.pkl")
EMBEDDING_FILE = os.path.join(UTILS_DIR, "embedding.npy")


# 可改参数
WORD_EMB_DIM = 300
MIN_WORD_FREQ = 2
MAX_VOCAB_SIZE = 50000


def resolve_glove_file(cli_glove_file=None, env=None):
    env = os.environ if env is None else env
    if cli_glove_file:
        return cli_glove_file
    if env.get("MIND_GLOVE_FILE"):
        return env["MIND_GLOVE_FILE"]
    return DEFAULT_GLOVE_FILE


GLOVE_FILE = resolve_glove_file()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Build MIND large utils files and embedding matrix."
    )
    parser.add_argument(
        "--glove-file",
        help=(
            "Path to glove.840B.300d.txt. "
            "Priority: --glove-file > MIND_GLOVE_FILE > dataset/glove default."
        ),
    )
    return parser.parse_args(argv)


def check_exists(path, hint=None):
    if not os.path.exists(path):
        message = f"文件不存在: {path}"
        if hint:
            message = f"{message}\n{hint}"
        raise FileNotFoundError(message)


def clean_and_tokenize(text: str):
    if text is None:
        return []
    text = text.lower()
    # 只保留字母数字和基本符号分割
    tokens = re.findall(r"[a-z0-9]+", text)
    return tokens


def iter_news_titles(news_file):
    """
    news.tsv 列格式通常是：
    news_id, category, subcategory, title, abstract, url, title_entities, abstract_entities
    """
    with open(news_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            title = parts[3]
            yield title


def build_word_dict(news_files, min_word_freq=2, max_vocab_size=50000):
    word_cnt = {}

    for news_file in news_files:
        print(f"统计词频: {news_file}")
        for title in tqdm(iter_news_titles(news_file)):
            tokens = clean_and_tokenize(title)
            for w in tokens:
                word_cnt[w] = word_cnt.get(w, 0) + 1

    # 过滤低频
    items = [(w, c) for w, c in word_cnt.items() if c >= min_word_freq]
    # 按频率降序
    items.sort(key=lambda x: (-x[1], x[0]))

    if max_vocab_size is not None:
        items = items[:max_vocab_size]

    # 0 留给 padding, 1 留给 unknown
    word_dict = {
        "<PAD>": 0,
        "<UNK>": 1,
    }

    for i, (w, _) in enumerate(items, start=2):
        word_dict[w] = i

    return word_dict


def iter_user_ids(behaviors_file):
    """
    behaviors.tsv 常见格式：
    impression_id, user_id, time, history, impressions
    """
    with open(behaviors_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            uid = parts[1].strip()
            if uid:
                yield uid


def build_user_dict(behaviors_files):
    user_dict = {
        "<UNK>": 0
    }
    next_idx = 1

    for behaviors_file in behaviors_files:
        print(f"统计用户: {behaviors_file}")
        for uid in tqdm(iter_user_ids(behaviors_file)):
            if uid not in user_dict:
                user_dict[uid] = next_idx
                next_idx += 1

    return user_dict


def load_glove_for_vocab(glove_file, word_dict, emb_dim=300):
    vocab_size = len(word_dict)
    embedding_matrix = np.random.normal(scale=0.1, size=(vocab_size, emb_dim)).astype(np.float32)
    embedding_matrix[word_dict["<PAD>"]] = np.zeros((emb_dim,), dtype=np.float32)

    found = 0

    print("开始从 GloVe 加载词向量，这一步会比较慢...")
    with open(glove_file, "r", encoding="utf-8") as f:
        for line in tqdm(f):
            values = line.rstrip().split(" ")
            word = values[0]

            if word not in word_dict:
                continue

            vec = values[1:]
            if len(vec) != emb_dim:
                continue

            embedding_matrix[word_dict[word]] = np.asarray(vec, dtype=np.float32)
            found += 1

    print(f"GloVe 命中词数: {found}/{len(word_dict)}")
    return embedding_matrix


def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def main(argv=None):
    args = parse_args(argv)
    glove_file = resolve_glove_file(args.glove_file)

    os.makedirs(UTILS_DIR, exist_ok=True)

    for path in [TRAIN_NEWS, DEV_NEWS, TRAIN_BEHAVIORS, DEV_BEHAVIORS]:
        check_exists(path)

    check_exists(
        glove_file,
        hint=(
            "请通过以下任一方式提供 GloVe 文件路径："
            "\n1. 命令行参数 --glove-file"
            "\n2. 环境变量 MIND_GLOVE_FILE"
            f"\n3. 默认路径 {DEFAULT_GLOVE_FILE}"
        ),
    )

    print(f"使用的 GloVe 文件: {glove_file}")

    print("==== 第1步：构建 word_dict（train+dev news） ====")
    word_dict = build_word_dict(
        news_files=[TRAIN_NEWS, DEV_NEWS],
        min_word_freq=MIN_WORD_FREQ,
        max_vocab_size=MAX_VOCAB_SIZE,
    )
    save_pickle(word_dict, WORD_DICT_FILE)
    print(f"word_dict 已保存: {WORD_DICT_FILE}")
    print(f"word_dict size: {len(word_dict)}")

    print("\n==== 第2步：构建 uid2index（train+dev behaviors） ====")
    user_dict = build_user_dict(
        behaviors_files=[TRAIN_BEHAVIORS, DEV_BEHAVIORS]
    )
    save_pickle(user_dict, USER_DICT_FILE)
    print(f"user_dict 已保存: {USER_DICT_FILE}")
    print(f"user_dict size: {len(user_dict)}")

    print("\n==== 第3步：构建 embedding.npy（按 word_dict 查 GloVe） ====")
    embedding_matrix = load_glove_for_vocab(
        glove_file=glove_file,
        word_dict=word_dict,
        emb_dim=WORD_EMB_DIM,
    )
    np.save(EMBEDDING_FILE, embedding_matrix)
    print(f"embedding 已保存: {EMBEDDING_FILE}")
    print(f"embedding shape: {embedding_matrix.shape}")

    print("\n全部完成。")


if __name__ == "__main__":
    main()