import numpy as np
import pandas as pd
from scipy.special import expit, softmax
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

# 1. シミュレーション設定
np.random.seed(123)
num_data = 5000  # ログデータのサイズ
num_items = 100  # 全アイテム数
slate_size = 10  # ランキングのサイズ
dim_context = 5  # ユーザー特徴量の次元
dim_item_feature = 5  # アイテム特徴量の次元

# --- シミュレーションの「神のみぞ知る」真のモデル ---

# アイテム特徴量ベクトル（本来は既知）
item_features = np.random.randn(num_items, dim_item_feature)

# 真の関連性モデルのパラメータ
true_relevance_theta = np.random.randn(dim_context, dim_item_feature)

# 真のポジションバイアス (上位ほど注目されやすい)
position_bias = 0.9 ** np.arange(slate_size)


def get_true_ctr(context, item_id, position):
    """真のクリック確率を返す (関連性 * ポジションバイアス)

    args:
        context: (dim_context,)
        item_id:
        position:
    """
    # (1, dim_context) @ (dim_context, dim_item_feature) @ (dim_item_feature, 1) -> (1,)
    relevance_score = context @ true_relevance_theta @ item_features[item_id].T
    relevance_prob = expit(relevance_score)  # R -> [0, 1]
    return relevance_prob * position_bias[position]


# --- 方策の定義 ---

# 行動方策 pi_b (データを収集した方策)
b_policy_theta = (
    true_relevance_theta * 0.7 + np.random.randn(dim_context, dim_item_feature) * 0.3
)

# 評価方策 pi_e (評価したい新しい方策, より真のモデルに近い)
e_policy_theta = (
    true_relevance_theta * 0.9 + np.random.randn(dim_context, dim_item_feature) * 0.1
)


def get_slate_and_propensities(context, policy_theta):
    """スコアに基づいてスレート(ランキング)と各アイテムの提示確率を生成
    args:
        context: (dim_context,)
        policy_theta: (dim_context, dim_item_feature)
    returns:
        slate: (slate_size,)
        propensities: (slate_size,)
    """
    # 全アイテムのスコアを計算
    # item_features: (num_items, dim_item_feature)
    scores = context @ policy_theta @ item_features.T  # (num_items)

    # スコアを確率に変換 (softmax)
    probs = softmax(scores)

    # 確率に基づいてスレートをサンプリング (非復元抽出)
    # 単なるサンプリングであり、確率が高い方からとるわけではない
    slate = np.random.choice(num_items, size=slate_size, replace=False, p=probs)

    # 各ポジションでの提示確率（傾向スコア）を計算
    # p(a,k|x) = p(a が k番目に選ばれる確率)
    # これは厳密には複雑だが、ここでは p(a|x) で代用する（一般的な簡略化）
    propensities = probs[slate]

    return slate, propensities


# 3. ログデータの生成
contexts = np.random.randn(num_data, dim_context)
log_records = []

print("ログデータを生成中...")
for i in tqdm(range(num_data)):
    context = contexts[i]

    # 行動方策でスレートと傾向スコアを生成
    slate, propensities = get_slate_and_propensities(context, b_policy_theta)

    rewards = []
    for k, item_id in enumerate(slate):
        ctr = get_true_ctr(context, item_id, k)
        reward = np.random.binomial(1, ctr)
        rewards.append(reward)

    log_records.append(
        {
            "context": context,
            "slate": slate,
            "reward": np.array(rewards),
            "propensity": propensities,
        }
    )


# 4. 報酬モデルの学習 (DM, DR用)
print("\n報酬予測モデルを学習中...")
# 学習データを作成: (context, item_id, position) -> reward
reward_model_train_data = []
for record in log_records:
    for k, item_id in enumerate(record["slate"]):
        features = np.concatenate(
            [
                record["context"],
                item_features[item_id],
                np.array([k]),  # positionを特徴量として追加
            ]
        )
        reward = record["reward"][k]
        reward_model_train_data.append(np.append(features, reward))

reward_model_df = pd.DataFrame(reward_model_train_data)
X_train_reward = reward_model_df.iloc[:, :-1]
y_train_reward = reward_model_df.iloc[:, -1]

reward_model = LogisticRegression(solver="saga", max_iter=100)
reward_model.fit(X_train_reward, y_train_reward)


def estimate_reward(context, item_id, position, model):
    """学習済みモデルで報酬を予測"""
    features = np.concatenate(
        [context, item_features[item_id], np.array([position])]
    ).reshape(1, -1)
    return model.predict_proba(features)[0, 1]


# 5. 各種OPE推定量の計算
V_hat = {}
dm_values = []
ips_values = []
dr_values = []

print("OPE推定量を計算中...")
for i in tqdm(range(num_data)):
    context = log_records[i]["context"]
    observed_slate = log_records[i]["slate"]
    observed_rewards = log_records[i]["reward"]
    propensities_b = log_records[i]["propensity"]

    # --- DM ---
    # 評価方策が生成するであろうランキングの期待値を計算
    # (dim_context) @ (dim_context, dim_item_feature) @ (dim_item_feature, num_items) -> (num_items)
    scores_e = context @ e_policy_theta @ item_features.T
    probs_e = softmax(scores_e)
    # 全アイテム・全ポジションの予測報酬を使って期待値を計算
    expected_slate_reward_dm = 0
    for k in range(slate_size):
        for item_id in range(num_items):
            # あるアイテムがk番目にきたときの報酬を推定している。
            # p(item_id, k | context, item_feature, {i}_{i = 1}^{num_item} \ item_id)を考えられる気がするけど、ランキング中の他のアイテムからの影響を周辺化した確率として出しているのか、あるいは他のアイテムに依存しない独立性の仮定をおいているのか。
            r_hat = estimate_reward(context, item_id, k, reward_model)
            expected_slate_reward_dm += probs_e[item_id] * r_hat
    dm_values.append(expected_slate_reward_dm)

    # --- IPS & DR ---
    # 評価方策における観測アイテムの提示確率を取得
    scores_e_obs = context @ e_policy_theta @ item_features.T  # (num_items)
    probs_e_obs = softmax(scores_e_obs)
    propensities_e = probs_e_obs[observed_slate]

    iw = propensities_e / propensities_b

    slate_ips = 0
    slate_dr = 0
    for k in range(slate_size):
        item_id = observed_slate[k]
        r = observed_rewards[k]

        # IPS項
        slate_ips += iw[k] * r

        # DR項
        r_hat = estimate_reward(context, item_id, k, reward_model)
        ips_part_dr = iw[k] * (r - r_hat)
        slate_dr += ips_part_dr

    slate_dr += expected_slate_reward_dm  # DRのDM部分を加算
    ips_values.append(slate_ips)
    dr_values.append(slate_dr)


V_hat["DM"] = np.mean(dm_values)
V_hat["IPS"] = np.mean(ips_values)
V_hat["DR"] = np.mean(dr_values)


# --- 参考：評価方策の真の値 ---
true_V_e_sum = 0
print("評価方策の真の価値を計算中...")
for i in tqdm(range(num_data)):  # 同じコンテキスト分布で期待値を計算
    context = contexts[i]
    scores_e = context @ e_policy_theta @ item_features.T  # (num_items)
    probs_e = softmax(scores_e)
    expected_reward_true = 0
    for k in range(slate_size):
        for item_id in range(num_items):
            true_ctr = get_true_ctr(context, item_id, k)
            expected_reward_true += probs_e[item_id] * true_ctr
    true_V_e_sum += expected_reward_true
true_V_e = true_V_e_sum / num_data

# 行動方策の真の値
true_V_b_sum = 0
for i in tqdm(range(num_data)):
    context = contexts[i]
    scores_b = context @ b_policy_theta @ item_features.T
    probs_b = softmax(scores_b)
    expected_reward_true = 0
    for k in range(slate_size):
        for item_id in range(num_items):
            true_ctr = get_true_ctr(context, item_id, k)
            expected_reward_true += probs_b[item_id] * true_ctr
    true_V_b_sum += expected_reward_true
true_V_b = true_V_b_sum / num_data


# 6. 結果の表示
print("\n--- Ranking Off-Policy Evaluation Results ---")
print(f"Slate Size: {slate_size}, Total Items: {num_items}")
print(f"Behavior Policy's True Value: {true_V_b:.4f}")
print(f"Evaluation Policy's True Value: {true_V_e:.4f} (This is the target value)")
print("---------------------------------------------")
for estimator, value in V_hat.items():
    print(f"{estimator:<5}: {value:.4f}  (Bias: {value - true_V_e:+.4f})")
