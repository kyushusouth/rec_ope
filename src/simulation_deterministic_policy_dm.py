import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from tqdm import tqdm


def sigmoid(x: np.ndarray) -> np.ndarray:
    """シグモイド関数"""
    return 1 / (1 + np.exp(-x))


def get_slates(
    num_data: int,
    slate_size: int,
    contexts: np.ndarray,
    policy_theta: np.ndarray,
    item_features: np.ndarray,
) -> np.ndarray:
    """方策によって選択されたスレートを取得する

    Args:
        num_data: データ数
        slate_size: スレートのアイテム数
        contexts: ユーザー特徴量 (num_data, dim_context)
        policy_theta: 方策のパラメータ (dim_context, dim_item_feature)
        item_features: アイテム特徴量 (num_items, dim_item_feature)

    Returns:
        slates: 方策によって選択されたスレート (num_data, slate_size)

    """
    slate_list = []
    for i in tqdm(range(num_data), desc="get slates"):
        context = contexts[i]
        scores = context @ policy_theta @ item_features.T
        slate = np.argsort(-scores)[:slate_size]
        slate_list.append(slate)
    slates = np.array(slate_list)
    return slates


def get_expected_reward_function(
    contexts: np.ndarray,
    item_features: np.ndarray,
    rng: np.random.Generator,
    degree: int,
    z_score: bool,
) -> np.ndarray:
    """各ラウンドのアイテムに対する期待報酬を与える関数を取得する

    Args:
        contexts: ユーザー特徴量 (n_rounds, dim_context)
        item_features: アイテム特徴量 (n_items, dim_item_feature)
        rng: 乱数ジェネレータ
        degree: polynomial featureの次数
        z_score: 期待報酬の値を標準化するか指定する論理値

    Returns:
        expected_reward: 期待報酬 (n_rounds, n_items)

    """
    poly = PolynomialFeatures(degree=degree)
    context_ = poly.fit_transform(contexts)
    action_context_ = poly.fit_transform(item_features)
    n_rounds, dim_context = context_.shape
    n_actions, dim_action_context = action_context_.shape

    context_coef_ = rng.uniform(-1, 1, size=dim_context)
    action_coef_ = rng.uniform(-1, 1, size=dim_action_context)
    context_action_coef_ = rng.uniform(-1, 1, size=(dim_context, dim_action_context))

    context_values = np.tile((context_ @ context_coef_)[:, None], (1, n_actions))
    action_values = np.tile((action_context_ @ action_coef_)[None, :], (n_rounds, 1))
    context_action_values = context_ @ context_action_coef_ @ action_context_.T

    expected_reward = context_values + action_values + context_action_values
    if z_score:
        expected_reward = (
            expected_reward - expected_reward.mean()
        ) / expected_reward.std()
    expected_reward = degree * expected_reward
    return expected_reward


def get_expected_reward_factual(
    slates: np.ndarray,
    contexts: np.ndarray,
    slate_size: int,
    expected_reward: np.ndarray,
) -> np.ndarray:
    """期待報酬関数から、behavior policyによって選択されたスレートに対する期待報酬を取得する

    Args:
        slates: behavior policyによって選択されたスレート (n_rounds, slate_size)
        contexts: ユーザー特徴量 (n_rounds, dim_context)
        slate_size: スレートサイズ
        expected_reward: 期待報酬関数

    Returns:
        expected_reward_factual: 期待報酬 (n_rounds, slate_size)

    """
    n_rounds = contexts.shape[0]
    expected_reward_factual = np.zeros((n_rounds, slate_size))
    for i in tqdm(range(n_rounds), desc="get expected_reward_factual"):
        expected_reward_factual[i, :] = expected_reward[i, slates[i]]

    expected_reward_factual = sigmoid(expected_reward_factual)
    return expected_reward_factual


def sample_reward_given_expected_reward_factual(
    expected_reward_factual: np.ndarray, slate_size: int, rng: np.random.Generator
):
    """期待報酬を用いてサンプリングを行い、報酬を取得する

    Args:
        expected_reward_factual: 期待報酬 (n_rounds, slate_size)
        slate_size: スレートサイズ
        rng: 乱数ジェネレータ

    Returns:
        reward: 観測報酬 (n_rounds, slate_size)

    """
    sampled_reward_list = []
    for pos_ in np.arange(slate_size):
        expected_reward_factual_at_position = expected_reward_factual[:, pos_]
        sampled_rewards_at_position = rng.binomial(
            1, expected_reward_factual_at_position
        )
        sampled_reward_list.append(sampled_rewards_at_position)
    reward = np.array(sampled_reward_list).T
    return reward


def get_true_policy_value(
    num_data: int, slate_size: int, expected_reward_factual: np.ndarray
) -> np.float64:
    """方策の真の価値を取得する

    Args:
        num_data: データ数
        slate_size: スレートサイズ
        expected_reward_factual: 期待報酬 (n_rounds, slate_size)

    Returns:
        true_policy_value: 真の方策の価値

    """
    true_policy_value = 0
    for i in tqdm(range(num_data), desc="get true policy value"):
        for k in range(slate_size):
            r = expected_reward_factual[i, k]
            r /= np.log2(k + 2)
            true_policy_value += r
    true_policy_value /= num_data
    return true_policy_value


def get_dm_value(
    num_data: int,
    slate_size: int,
    contexts: np.ndarray,
    item_features: np.ndarray,
    slates: np.ndarray,
    reward_model: LogisticRegression,
) -> np.float64:
    """DM推定量を取得する

    Args:
        num_data: データ数
        slate_size: スレートサイズ
        contexts: ユーザー特徴量 (n_rounds, dim_context)
        item_features: アイテム特徴量 (n_items, dim_item_feature)
        slates: スレート (n_rounds, slate_size)
        reward_model: 報酬予測モデル

    Returns:
        dm_value: DM推定量

    """
    dm_value = 0
    for i in tqdm(range(num_data), desc="get dm_value", leave=False):
        context = contexts[i]
        for k in range(slate_size):
            item_feature = item_features[slates[i, k]]
            r_hat = reward_model.predict_proba(
                np.concatenate([context, item_feature]).reshape(1, -1)
            )[0][1]
            r_hat /= np.log2(k + 2)
            dm_value += r_hat
    dm_value /= num_data
    return dm_value


def main():
    num_simulations = 50
    num_data = 10000
    num_items = 1000
    slate_size = 10
    dim_context = 16
    dim_item_feature = 16
    noise_b = 0.5
    noise_e = 0.3

    rng_true = np.random.default_rng(num_simulations + 1)
    contexts = rng_true.normal(0, 1, (num_data, dim_context))
    item_features = rng_true.normal(0, 1, (num_items, dim_item_feature))
    true_relevance_theta = rng_true.normal(0, 1, (dim_context, dim_item_feature))

    b_policy_theta = true_relevance_theta + rng_true.normal(
        0, noise_b, true_relevance_theta.shape
    )
    e_policy_theta = true_relevance_theta + rng_true.normal(
        0, noise_e, true_relevance_theta.shape
    )

    slates_b = get_slates(num_data, slate_size, contexts, b_policy_theta, item_features)
    slates_e = get_slates(num_data, slate_size, contexts, e_policy_theta, item_features)

    expected_reward = get_expected_reward_function(
        contexts, item_features, rng_true, 1, True
    )

    expected_reward_factual_b = get_expected_reward_factual(
        slates_b, contexts, slate_size, expected_reward
    )
    expected_reward_factual_e = get_expected_reward_factual(
        slates_e, contexts, slate_size, expected_reward
    )

    true_b_policy_value = get_true_policy_value(
        num_data, slate_size, expected_reward_factual_b
    )
    true_e_policy_value = get_true_policy_value(
        num_data, slate_size, expected_reward_factual_e
    )

    results = []
    for seed in tqdm(range(num_simulations), desc="simulation"):
        rng = np.random.default_rng(seed)

        sampled_reward = sample_reward_given_expected_reward_factual(
            expected_reward_factual_b, slate_size, rng
        )

        reward_model_train_data = []
        for i in tqdm(
            range(num_data), desc="get reward_model training data", leave=False
        ):
            context = contexts[i]
            for k in range(slate_size):
                item_feature = item_features[slates_b[i, k]]
                reward = np.array([sampled_reward[i, k]])
                reward_model_train_data.append(
                    np.concatenate([context, item_feature, reward])
                )
        reward_model_df = pd.DataFrame(reward_model_train_data)
        X_train_reward = reward_model_df.iloc[:, :-1]
        y_train_reward = reward_model_df.iloc[:, -1]
        reward_model = LogisticRegression(random_state=seed)
        reward_model.fit(X_train_reward, y_train_reward)

        dm_value_b = get_dm_value(
            num_data, slate_size, contexts, item_features, slates_b, reward_model
        )
        dm_value_e = get_dm_value(
            num_data, slate_size, contexts, item_features, slates_e, reward_model
        )

        result_df = pd.DataFrame(
            [
                {
                    "estimator": "dm",
                    "pred_value_b": dm_value_b,
                    "pred_value_e": dm_value_e,
                }
            ]
        )
        result_df["seed"] = seed
        results.append(result_df)

    results_df = pd.concat(results, axis=0)
    results_df = results_df.assign(
        squared_error=lambda x: (x["pred_value_e"] - true_e_policy_value) ** 2,
        is_wrong_choice=lambda x: np.where(
            true_e_policy_value > true_b_policy_value,
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
        .assign(
            squared_bias=lambda x: (x["pred_value_e_mean"] - true_e_policy_value) ** 2
        )
        .assign(variance=lambda x: x["mse"] - x["squared_bias"])
    )

    print(results_agg_df)


if __name__ == "__main__":
    main()
