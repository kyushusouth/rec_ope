from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.special import expit, softmax
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

sns.set_style("whitegrid")


def get_slate_and_pscore(
    context: np.ndarray,
    policy_theta: np.ndarray,
    item_features: np.ndarray,
    rng: np.random.Generator,
    item_ids: np.ndarray,
    slate_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """スレートと各アイテムの傾向スコアを取得

    Args:
        context: ユーザー特徴量 (dim_context,)
        policy_theta: 方策のパラメータ (dim_context, dim_item_feature)
        item_features: アイテム特徴量 (num_items, dim_item_feature)
        rng: 乱数ジェネレータ
        num_items: アイテム数
        slate_size: スレートに含まれるアイテム数

    Returns:
        slate: 選ばれたスレート (slate_size,)
        pscores: 傾向スコア (slate_size,)
    """
    scores = context @ policy_theta @ item_features.T
    probs = softmax(scores)
    slate = rng.choice(item_ids, (slate_size), replace=True, p=probs)
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


def estimate_reward(
    context: np.ndarray,
    item_feature: np.ndarray,
    position: int,
    model: LogisticRegression,
) -> np.ndarray:
    """報酬を推定する

    Args:
        context: ユーザー特徴量 (dim_context,)
        item_feature: アイテム特徴量 (dim_item_feature,)
        position: スレートにおける位置
        model: 報酬推定モデル

    Returns
        pred: 予測された報酬
    """
    features = np.concatenate([context, item_feature, np.array([position])]).reshape(
        1, -1
    )
    pred = model.predict_proba(features)[0][1]
    return pred


def calculate_true_policy_value(
    policy_name: str,
    num_data: int,
    contexts: np.ndarray,
    policy_theta: np.ndarray,
    item_features: np.ndarray,
    slate_size: int,
    num_items: int,
    true_relevance_theta: np.ndarray,
    position_bias: np.ndarray,
) -> np.float64:
    """方策の真の価値を計算する

    Args:
        policy_name: 方策の名前
        num_data: ログデータの数
        contexts: ユーザー特徴量 (num_data, dim_context)
        policy_theta: 方策のパラメータ (dim_context, dim_item_feature)
        item_features: アイテム特徴量 (num_items, dim_item_feature)
        slate_size: スレートに含まれるアイテム数
        num_items: アイテム数
        true_relevance_theta: ユーザーとアイテムの真の関連性 (dim_context, dim_item_feature)
        position_bias: 位置バイアスを考慮する係数ベクトル (slate_size,)

    Returns:
        value: 方策の価値
    """
    value_sum = 0
    for i in tqdm(
        range(num_data),
        desc=f"calculate {policy_name} policy's true value",
        leave=False,
    ):
        context = contexts[i]
        scores = context @ policy_theta @ item_features.T  # (num_items)
        probs = softmax(scores)
        expected_reward_true = 0
        for k in range(slate_size):
            for item_id in range(num_items):
                true_cvr = get_true_cvr(
                    context,
                    true_relevance_theta,
                    item_features,
                    item_id,
                    position_bias,
                    k,
                )
                expected_reward_true += probs[item_id] * true_cvr
        value_sum += expected_reward_true
    value = value_sum / num_data
    return value


def estimate_policy_value(
    num_data: int,
    log_records: list[dict[str, np.ndarray]],
    e_policy_theta: np.ndarray,
    item_features: np.ndarray,
    slate_size: int,
    num_items: int,
    reward_model: LogisticRegression,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """方策の価値を推定する

    Args:
        num_data: ログデータの数
        log_records: ログデータ (num_data,)
        e_policy_theta: 評価方策のパラメータ
        item_features: アイテム特徴量 (num_items, dim_item_feature)
        slate_size: スレートに含まれるアイテム数
        num_items: アイテム数
        reward_model: 報酬推定モデル

    Returns:
        dm_values: DM推定量 (num_data,)
        ips_values: IPS推定量 (num_data,)
        dr_values: DR推定量 (num_data,)
    """
    dm_values = []
    ips_values = []
    dr_values = []

    for i in tqdm(
        range(num_data), desc="estimate evaluation policy's value", leave=False
    ):
        context = log_records[i]["context"]
        observed_slate = log_records[i]["slate"]
        observed_rewards = log_records[i]["reward"]
        pscores_b = log_records[i]["pscore"]

        scores_e = context @ e_policy_theta @ item_features.T  # (num_items,)
        probs_e = softmax(scores_e)
        expected_slate_reward_dm = 0
        for k in range(slate_size):
            for item_id in range(num_items):
                r_hat = estimate_reward(
                    context, item_features[item_id], k, reward_model
                )
                expected_slate_reward_dm += probs_e[item_id] * r_hat
        dm_values.append(expected_slate_reward_dm)

        pscores_e = probs_e[observed_slate]
        iw = pscores_e / pscores_b
        slate_ips = 0
        slate_dr = 0
        for k in range(slate_size):
            item_id = observed_slate[k]
            r = observed_rewards[k]
            slate_ips += iw[k] * r
            r_hat = estimate_reward(context, item_features[item_id], k, reward_model)
            ips_part_dr = iw[k] * (r - r_hat)
            slate_dr += ips_part_dr
        slate_dr += expected_slate_reward_dm
        ips_values.append(slate_ips)
        dr_values.append(slate_dr)

    return dm_values, ips_values, dr_values


def main():
    num_simulation = 10
    num_data = 100
    num_items = 100
    slate_size = 10
    dim_context = 10
    dim_item_feature = 10
    root_dir = Path(__file__).parent.parent
    result_dir = root_dir.joinpath("result", datetime.now().strftime("%Y%m%d-%H%M%S"))
    result_dir.mkdir(parents=True, exist_ok=True)

    rng_fix = np.random.default_rng(num_simulation)
    item_ids = np.arange(num_items)
    item_features = rng_fix.normal(0, 1, (num_items, dim_item_feature))
    true_relevance_theta = rng_fix.normal(0, 1, (dim_context, dim_item_feature))
    position_bias = 0.9 ** np.arange(slate_size)
    b_policy_theta = (
        true_relevance_theta * 0.7
        + rng_fix.normal(0, 1, (dim_context, dim_item_feature)) * 0.3
    )
    e_policy_theta = (
        true_relevance_theta * 0.9
        + rng_fix.normal(0, 1, (dim_context, dim_item_feature)) * 0.1
    )
    contexts = rng_fix.normal(0, 1, (num_data, dim_context))

    true_V_e = calculate_true_policy_value(
        "evaluation",
        num_data,
        contexts,
        e_policy_theta,
        item_features,
        slate_size,
        num_items,
        true_relevance_theta,
        position_bias,
    )
    true_V_b = calculate_true_policy_value(
        "behaivior",
        num_data,
        contexts,
        b_policy_theta,
        item_features,
        slate_size,
        num_items,
        true_relevance_theta,
        position_bias,
    )

    results = []

    for seed in tqdm(range(num_simulation), desc="simulation"):
        rng = np.random.default_rng(seed)
        log_records = []

        for i in tqdm(range(num_data), desc="create log_records", leave=False):
            context = contexts[i]
            slate, pscores = get_slate_and_pscore(
                context, b_policy_theta, item_features, rng, item_ids, slate_size
            )
            rewards = []
            for k, item_id in enumerate(slate):
                cvr = get_true_cvr(
                    context,
                    true_relevance_theta,
                    item_features,
                    item_id,
                    position_bias,
                    k,
                )
                reward = rng.binomial(1, cvr)
                rewards.append(reward)
            log_records.append(
                {
                    "context": context,
                    "slate": slate,
                    "reward": rewards,
                    "pscore": pscores,
                }
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

        dm_values_e, ips_values_e, dr_values_e = estimate_policy_value(
            num_data,
            log_records,
            e_policy_theta,
            item_features,
            slate_size,
            num_items,
            reward_model,
        )
        dm_values_b, ips_values_b, dr_values_b = estimate_policy_value(
            num_data,
            log_records,
            b_policy_theta,
            item_features,
            slate_size,
            num_items,
            reward_model,
        )

        result_df = pd.DataFrame(
            [
                {
                    "estimator": "dm",
                    "pred_value_e": np.mean(dm_values_e),
                    "pred_value_b": np.mean(dm_values_b),
                },
                {
                    "estimator": "ips",
                    "pred_value_e": np.mean(ips_values_e),
                    "pred_value_b": np.mean(ips_values_b),
                },
                {
                    "estimator": "dr",
                    "pred_value_e": np.mean(dr_values_e),
                    "pred_value_b": np.mean(dr_values_b),
                },
            ]
        )
        result_df["seed"] = seed
        results.append(result_df)

    results_df = pd.concat(results, axis=0)
    results_df = results_df.assign(
        squared_error=lambda x: (x["pred_value_e"] - true_V_e) ** 2,
        is_wrong_choice=lambda x: np.where(
            true_V_e > true_V_b,
            np.where(x["pred_value_e"] <= x["pred_value_b"], 1, 0),
            np.where(x["pred_value_e"] > x["pred_value_b"], 1, 0),
        ),
    )
    results_agg_df = (
        results_df.groupby("estimator")
        .agg(
            pred_value_e_mean=("pred_value_e", "mean"),
            mse=("squared_error", "mean"),
            wrong_choice_rate=("is_wrong_choice", "mean"),
        )
        .reset_index()
        .assign(squared_bias=lambda x: (x["pred_value_e_mean"] - true_V_e) ** 2)
        .assign(variance=lambda x: x["mse"] - x["squared_bias"])
    )

    plt.figure(figsize=(12, 8))
    sns.barplot(data=results_agg_df, y="estimator", x="squared_bias")
    plt.tight_layout()
    plt.savefig(result_dir.joinpath("squared_bias.png"))

    plt.figure(figsize=(12, 8))
    sns.barplot(data=results_agg_df, y="estimator", x="variance")
    plt.tight_layout()
    plt.savefig(result_dir.joinpath("variance.png"))

    plt.figure(figsize=(12, 8))
    sns.barplot(data=results_agg_df, y="estimator", x="mse")
    plt.tight_layout()
    plt.savefig(result_dir.joinpath("mse.png"))

    plt.figure(figsize=(12, 8))
    sns.barplot(data=results_agg_df, y="estimator", x="wrong_choice_rate")
    plt.tight_layout()
    plt.savefig(result_dir.joinpath("wrong_choice_rate.png"))


if __name__ == "__main__":
    main()
