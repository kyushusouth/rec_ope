import numpy as np
import pandas as pd
from scipy.special import expit, softmax
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm


def get_slate_and_pscore(
    context: np.ndarray,
    policy_theta: np.ndarray,
    item_features: np.ndarray,
    rng: np.random.Generator,
    num_items: int,
    slate_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """スレートと各アイテムの傾向スコアを取得

    Args:
        context: ユーザー特徴量 (dim_context,)
        policy_theta: 方策のパラメータ (dim_context, dim_item_feature)
        item_features: アイテム特徴量 (num_items, dim_item_feature)
        rng: 乱数ジェネレータ
        num_items: アイテム数
        slate_size: スレートの数

    Returns:
        slate: 選ばれたスレート (slate_size,)
        pscores: 傾向スコア (slate_size,)
    """
    scores = context @ policy_theta @ item_features.T
    probs = softmax(scores)
    slate = rng.integers(0, num_items, slate_size)
    pscores = probs[slate]
    return slate, pscores


def get_true_cvr(
    context: np.ndarray,
    true_relevance: np.ndarray,
    item_features: np.ndarray,
    item_id: str,
    position_bias: np.ndarray,
    position: int,
) -> np.float64:
    """真のCVRを取得する

    Args:
        context: ユーザー特徴量 (dim_context,)
        true_relevance: ユーザーとアイテムの真の関連性 (dim_context, dim_item_feature)
        item_features: アイテムの特徴量 (num_items, dim_item_feature)
        item_id: アイテムID
        position_bias: スレートにおける位置バイアスを考慮する係数ベクトル (slate_size,)
        position: スレートの位置

    Returns:
        relevance_prob: CVR (num_items,)
    """
    relevance_score = context @ true_relevance @ item_features[item_id].T
    relevance_prob = expit(relevance_score)
    relevance_prob = relevance_prob * position_bias[position]
    return relevance_prob


def main():
    seed = 42
    num_data = 10000
    num_items = 1000
    slate_size = 10
    dim_context = 10
    dim_item_feature = 10

    rng = np.random.default_rng(seed)

    item_features = rng.normal(0, 1, (num_items, dim_item_feature))
    true_relevance_theta = rng.normal(0, 1, (dim_context, dim_item_feature))
    position_bias = 0.9 ** np.arange(slate_size)

    b_policy_theta = (
        true_relevance_theta * 0.7
        + rng.normal(0, 1, (dim_context, dim_item_feature)) * 0.3
    )
    e_policy_theta = (
        true_relevance_theta * 0.9
        + rng.normal(0, 1, (dim_context, dim_item_feature)) * 0.1
    )

    contexts = rng.normal(0, 1, (num_data, dim_context))
    log_records = []

    for i in tqdm(range(num_data)):
        context = contexts[i]
        slate, pscores = get_slate_and_pscore(
            context, b_policy_theta, item_features, rng, num_items, slate_size
        )
        rewards = []
        for k, item_id in enumerate(slate):
            cvr = get_true_cvr(
                context, true_relevance_theta, item_features, item_id, position_bias, k
            )
            reward = rng.binomial(1, cvr)
            rewards.append(reward)
        log_records.append(
            {"context": context, "slate": slate, "reward": rewards, "pscore": pscores}
        )

    reward_model_train_data = []
    for record in log_records:
        for k, item_id in enumerate(record["slate"]):
            features = np.concatenate(
                [record["context"], item_features[item_id], np.array([k])], axis=0
            )
            reward = record["reward"][k]
            reward_model_train_data.append(np.append(features, reward))

    reward_model_df = pd.DataFrame(reward_model_train_data)
    X_train_reward = reward_model_df.iloc[:, :-1]
    y_train_reward = reward_model_df.iloc[:, -1]
    reward_model = LogisticRegression(random_state=seed)
    reward_model.fit(X_train_reward, y_train_reward)


if __name__ == "__main__":
    main()
