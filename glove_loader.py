from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parent


def default_logger(message):
    print(message, flush=True)


def resolve_glove_text_path(embedding_dim, cache_dir="glove"):
    cache_path = Path(cache_dir)
    if not cache_path.is_absolute():
        cache_path = REPO_ROOT / cache_path
    if embedding_dim == 300:
        return cache_path / "glove.840B.300d.txt"
    return cache_path / f"glove.6B.{embedding_dim}d.txt"


def load_glove_lookup(word_dict, embedding_dim, cache_dir="glove", glove_cls=None, logger=None, progress_interval=500000):
    if glove_cls is None:
        try:
            from torchtext.vocab import GloVe as torchtext_glove
        except Exception:
            torchtext_glove = None
        glove_cls = torchtext_glove

    if glove_cls is not None:
        try:
            glove_name = "840B" if embedding_dim == 300 else "6B"
            if logger is not None:
                logger(f"Loading GloVe via torchtext ({glove_name}, dim={embedding_dim})")
            glove = glove_cls(name=glove_name, dim=embedding_dim, cache=cache_dir, max_vectors=10000000000)
            if logger is not None:
                logger(f"Loaded GloVe via torchtext with {len(glove.stoi)} entries")
            return glove.stoi, glove.vectors, torch.mean(glove.vectors, dim=0, keepdim=False)
        except Exception as exc:
            if logger is not None:
                logger(f"torchtext GloVe unavailable ({exc}); falling back to text scan")

    return load_glove_lookup_from_text(
        word_dict,
        embedding_dim,
        cache_dir,
        logger=logger,
        progress_interval=progress_interval,
    )


def load_glove_lookup_from_text(word_dict, embedding_dim, cache_dir="glove", logger=None, progress_interval=500000):
    glove_path = resolve_glove_text_path(embedding_dim, cache_dir)
    if not glove_path.exists():
        raise FileNotFoundError(f"GloVe text file not found: {glove_path}")

    target_words = {word for word in word_dict if word not in {"<PAD>", "<UNK>"}}
    vectors = {}
    sum_vector = torch.zeros(embedding_dim, dtype=torch.float32)
    row_count = 0

    if logger is not None:
        logger(f"Scanning GloVe text file: {glove_path}")

    with open(glove_path, "r", encoding="utf-8") as glove_file:
        for line in glove_file:
            parts = line.rstrip("\n").split(" ")
            if len(parts) != embedding_dim + 1:
                continue
            word = parts[0]
            values = torch.tensor([float(value) for value in parts[1:]], dtype=torch.float32)
            sum_vector += values
            row_count += 1
            if word in target_words and word not in vectors:
                vectors[word] = values
            if logger is not None and progress_interval and row_count % progress_interval == 0:
                logger(f"GloVe scan progress: processed {row_count} rows, matched {len(vectors)}/{len(target_words)} target words")

    if row_count == 0:
        raise ValueError(f"No usable GloVe rows found in {glove_path}")

    if logger is not None:
        logger(f"Finished scanning GloVe text file: {row_count} rows, matched {len(vectors)}/{len(target_words)} target words")

    return vectors, None, sum_vector / row_count


def build_word_embedding_vectors(word_dict, embedding_dim, cache_dir="glove", glove_cls=None, logger=None, progress_interval=500000):
    glove_lookup, glove_vectors, glove_mean_vector = load_glove_lookup(
        word_dict=word_dict,
        embedding_dim=embedding_dim,
        cache_dir=cache_dir,
        glove_cls=glove_cls,
        logger=logger,
        progress_interval=progress_interval,
    )
    word_embedding_vectors = torch.zeros([len(word_dict), embedding_dim])
    for word, index in word_dict.items():
        if index == 0:
            continue
        if glove_vectors is not None and word in glove_lookup:
            word_embedding_vectors[index, :] = glove_vectors[glove_lookup[word]]
            continue
        if glove_vectors is None and word in glove_lookup:
            word_embedding_vectors[index, :] = glove_lookup[word]
            continue
        random_vector = torch.zeros(embedding_dim)
        random_vector.normal_(mean=0, std=0.1)
        word_embedding_vectors[index, :] = random_vector + glove_mean_vector
    if logger is not None:
        logger(f"Built word embedding matrix with shape {tuple(word_embedding_vectors.shape)}")
    return word_embedding_vectors
