import os
import re
import pickle
import argparse
from collections import Counter
import numpy as np
from pathlib import Path


PAT = re.compile(r"[\w]+|[.,!?;|]")


def get_default_base_dir():
    return os.environ.get("MIND_BASE_DIR", str(Path(__file__).resolve().parent))


def word_tokenize(sent: str):
    if isinstance(sent, str):
        return PAT.findall(sent.lower())
    return []


def read_news_titles(news_file):
    with open(news_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            # MIND: nid, category, subcategory, title, abstract, url, title_entities, abstract_entities
            title = parts[3]
            yield title


def read_user_ids(behaviors_file):
    with open(behaviors_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            # MIND: impression_id, user_id, time, history, impressions
            uid = parts[1]
            yield uid


def build_word_dict(news_files, min_freq=1, max_vocab=None):
    counter = Counter()

    for news_file in news_files:
        print(f"[word_dict] reading titles from: {news_file}")
        for title in read_news_titles(news_file):
            counter.update(word_tokenize(title))

    words = [w for w, c in counter.items() if c >= min_freq]
    words.sort(key=lambda w: (-counter[w], w))

    if max_vocab is not None:
        words = words[:max_vocab]

    # 0 预留给 PAD / UNK
    word_dict = {w: i + 1 for i, w in enumerate(words)}

    print(f"[word_dict] vocab size (without 0): {len(word_dict)}")
    return word_dict, counter


def build_user_dict(behaviors_files):
    seen = set()
    users = []

    for behaviors_file in behaviors_files:
        print(f"[uid2index] reading users from: {behaviors_file}")
        for uid in read_user_ids(behaviors_file):
            if uid not in seen:
                seen.add(uid)
                users.append(uid)

    # 0 预留给 unknown user
    uid2index = {uid: i + 1 for i, uid in enumerate(users)}
    print(f"[uid2index] user count (without 0): {len(uid2index)}")
    return uid2index


def load_glove_for_vocab(glove_file, vocab_set, emb_dim=300):
    """
    只加载 vocab_set 中需要的词，避免把整个 GloVe 全读进内存。
    """
    vectors = {}
    total = 0

    print(f"[glove] scanning: {glove_file}")
    with open(glove_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            total += 1
            line = line.rstrip("\n")
            if not line:
                continue

            parts = line.split(" ")
            if len(parts) < emb_dim + 1:
                continue

            # 更稳一点：从右边取 300 维，左边拼回 token
            word = " ".join(parts[:-emb_dim])
            if word not in vocab_set:
                continue

            try:
                vec = np.asarray(parts[-emb_dim:], dtype=np.float32)
            except ValueError:
                continue

            if vec.shape[0] != emb_dim:
                continue

            if word not in vectors:
                vectors[word] = vec

    print(f"[glove] matched {len(vectors)} / {len(vocab_set)} words")
    return vectors


def build_embedding_matrix(word_dict, glove_vectors, emb_dim=300, seed=42):
    rng = np.random.default_rng(seed)

    # [vocab_size + 1, emb_dim], 第 0 行给 PAD / UNK
    emb = rng.normal(loc=0.0, scale=0.1, size=(len(word_dict) + 1, emb_dim)).astype(np.float32)
    emb[0] = 0.0

    hit = 0
    for word, idx in word_dict.items():
        vec = glove_vectors.get(word)
        if vec is not None:
            emb[idx] = vec
            hit += 1

    print(f"[embedding] coverage: {hit}/{len(word_dict)} = {hit / max(len(word_dict), 1):.4f}")
    print(f"[embedding] shape: {emb.shape}")
    return emb


def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default=get_default_base_dir())
    parser.add_argument("--glove_file", type=str, required=True)
    parser.add_argument("--include_dev", action="store_true")
    parser.add_argument("--min_freq", type=int, default=1)
    parser.add_argument("--max_vocab", type=int, default=None)
    parser.add_argument("--emb_dim", type=int, default=300)
    args = parser.parse_args()

    data_dir = os.path.join(args.base_dir, "dataset")
    train_news = os.path.join(data_dir, "MINDsmall_train", "news.tsv")
    train_behaviors = os.path.join(data_dir, "MINDsmall_train", "behaviors.tsv")
    dev_news = os.path.join(data_dir, "MINDsmall_dev", "news.tsv")
    dev_behaviors = os.path.join(data_dir, "MINDsmall_dev", "behaviors.tsv")
    utils_dir = os.path.join(data_dir, "utils")

    os.makedirs(utils_dir, exist_ok=True)

    for p in [train_news, train_behaviors, args.glove_file]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"not found: {p}")

    news_files = [train_news]
    behavior_files = [train_behaviors]

    if args.include_dev:
        if not os.path.exists(dev_news) or not os.path.exists(dev_behaviors):
            raise FileNotFoundError("include_dev=True, but dev files not found")
        news_files.append(dev_news)
        behavior_files.append(dev_behaviors)

    print("=== Step 1: build word_dict.pkl ===")
    word_dict, counter = build_word_dict(
        news_files=news_files,
        min_freq=args.min_freq,
        max_vocab=args.max_vocab,
    )

    print("=== Step 2: build uid2index.pkl ===")
    uid2index = build_user_dict(behavior_files)

    print("=== Step 3: load needed glove vectors ===")
    glove_vectors = load_glove_for_vocab(
        glove_file=args.glove_file,
        vocab_set=set(word_dict.keys()),
        emb_dim=args.emb_dim,
    )

    print("=== Step 4: build embedding.npy ===")
    embedding = build_embedding_matrix(
        word_dict=word_dict,
        glove_vectors=glove_vectors,
        emb_dim=args.emb_dim,
    )

    word_dict_path = os.path.join(utils_dir, "word_dict.pkl")
    uid2index_path = os.path.join(utils_dir, "uid2index.pkl")
    embedding_path = os.path.join(utils_dir, "embedding.npy")

    save_pickle(word_dict, word_dict_path)
    save_pickle(uid2index, uid2index_path)
    np.save(embedding_path, embedding)

    print("\nDone.")
    print(f"saved: {word_dict_path}")
    print(f"saved: {uid2index_path}")
    print(f"saved: {embedding_path}")


if __name__ == "__main__":
    main()
