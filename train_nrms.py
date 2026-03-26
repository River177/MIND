import os
import pickle
import tensorflow as tf

from recommenders.models.newsrec.io.mind_iterator import MINDIterator
from recommenders.models.newsrec.models.nrms import NRMSModel
from recommenders.models.newsrec.newsrec_utils import prepare_hparams


BASE_DIR = "/home/njvivo/wuhao/MIND"
DATA_DIR = os.path.join(BASE_DIR, "dataset")
UTILS_DIR = os.path.join(DATA_DIR, "utils")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

TRAIN_DIR = os.path.join(DATA_DIR, "MINDsmall_train")
DEV_DIR = os.path.join(DATA_DIR, "MINDsmall_dev")

train_news_file = os.path.join(TRAIN_DIR, "news.tsv")
train_behaviors_file = os.path.join(TRAIN_DIR, "behaviors.tsv")
valid_news_file = os.path.join(DEV_DIR, "news.tsv")
valid_behaviors_file = os.path.join(DEV_DIR, "behaviors.tsv")

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
        title_size=30,      # 标题最大长度
        his_size=50,        # 用户历史点击数
        npratio=4,          # 负采样数
        word_emb_dim=300,   # 必须和你的 GloVe 维度一致

        # NRMS 模型结构
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


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. 检查文件
    for path in [
        train_news_file,
        train_behaviors_file,
        valid_news_file,
        valid_behaviors_file,
        wordEmb_file,
        wordDict_file,
        userDict_file,
    ]:
        check_exists(path)

    # 2. GPU
    setup_gpu()

    # 3. 打印 utils 信息
    show_basic_info()

    # 4. 构建超参数
    hparams = build_hparams()

    # 5. 构建模型
    print("开始构建 NRMS 模型...")
    model = NRMSModel(hparams, MINDIterator, seed=42)

    # 6. 训练
    print("开始训练...")
    model.fit(
        train_news_file,
        train_behaviors_file,
        valid_news_file,
        valid_behaviors_file,
    )

    # 7. 验证集评估
    print("开始验证集评估...")
    res = model.run_eval(valid_news_file, valid_behaviors_file)
    print("验证结果:", res)

    # 8. 保存权重
    weight_path = os.path.join(OUTPUT_DIR, "nrms_weights.h5")
    model.model.save_weights(weight_path)
    print("模型权重已保存到:", weight_path)


if __name__ == "__main__":
    main()