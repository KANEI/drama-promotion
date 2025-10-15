import os
import pymc as pm
import arviz as az
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib

# データフレーム作成
## 現在のディレクトリのパスを取得
current_directory = os.getcwd()
file_path = os.path.join(current_directory,"data","過去の本企画集客.csv")

## データフレーム化
df = pd.read_csv(file_path)

## グループ化
df_stage = df[["企画", "キャパ", "予約数"]].groupby("企画").sum()

df_count = df[["企画","キャパ"]].groupby(["企画"]).count()
df_count = df_count.rename(columns={"キャパ":"公演数"})

### 有料公演がTRUE(1)の時、全て足したらステ数と一致する。
### これを利用して有料公演のダミー変数を作成。
df_price = df[["企画","有料公演"]].groupby(["企画"]).sum()
df_price["有料公演"] = (df_price["有料公演"]==df_count["公演数"]).astype(int)

merged_df = df_stage.merge(df_count, on='企画')
merged_df = merged_df.merge(df_price, on='企画')

print("----- データ -----")
print(merged_df.head())
print("-----------------")
print(merged_df.describe())
print("-----------------")
stage_count = len(merged_df)
print(f"データ数：{stage_count}")
print("-----------------")
## ヒストグラムを作成
ax = merged_df.hist()
## 保存（ファイルパスを指定）
plt.savefig("outputs/histogram.png") 


# MCMCを実行する。
print("----- MCMCを実行 -----")

## データセットの作成
N = merged_df['キャパ'].values
X = merged_df['予約数'].values
Y = merged_df['公演数'].values
Z = merged_df['有料公演'].values

with pm.Model() as logistic_model:
    ## 事前分布を設定。βは独立な標準正規分布から生成されるとする。
    beta_0 = pm.Normal('beta_0', mu=0, sigma=1)
    beta_1 = pm.Normal('beta_1', mu=0, sigma=1)
    beta_2 = pm.Normal('beta_2', mu=0, sigma=1)

    ## ロジットリンク関数
    eta = beta_0 + beta_1 * Y + beta_2 * Z
    p = pm.invlogit(eta)  # pm.invlogit は expit と同じ

    ## 尤度関数
    ### 二項分布の尤度
    X_obs = pm.Binomial('X_obs', n=N, p=p, observed=X)

    ## MCMCサンプリングの実行
    trace = pm.sample(5000, tune=1000, cores=2, random_seed=42)

## βの事後分布の分析
print("### 1. パラメータ(beta)の事後分布の要約 ###")
print(az.summary(trace, var_names=['beta_0', 'beta_1', 'beta_2']))

# 新しい条件での p の事後サンプルを計算
# Y = 4, Z = 1(ステ4回, 有料公演)

## サンプリングされたパラメータの事後サンプルを抽出
beta_0_samples = trace.posterior['beta_0'].values.flatten()
beta_1_samples = trace.posterior['beta_1'].values.flatten()
beta_2_samples = trace.posterior['beta_2'].values.flatten()

# 新しい条件 (Y=4, Z=1) で eta を計算
eta_new_samples = beta_0_samples + beta_1_samples * 4 + beta_2_samples * 1

# 新しい条件での p の事後サンプルを計算
p_new_samples = 1 / (1 + np.exp(-eta_new_samples))

# 事後平均を計算
posterior_mean = np.mean(p_new_samples)

# HDIを計算
hdi_interval = az.hdi(p_new_samples, hdi_prob=0.94)

print(f"Y=4, Z=1 の場合の p の事後平均: {posterior_mean:.4f}")
print(f"Y=4, Z=1 の場合の p の94% HDI区間: {hdi_interval}")

# 最終的な動員予測
print(f"--------------------")
print(f"キャパ38で、4ステ料金制の公演とする。")
print(f"1ステあたりの予測動員数: {38*posterior_mean}")
print(f"予測総動員数: {152*posterior_mean}")
print(f"1ステあたりの動員数の94%予測区間: {38*hdi_interval}")
print(f"総動員数の94%予測区間: {152*hdi_interval}")
print(f"--------------------")

# テキストに出力
text = f"-------- 過去の公演の基本的な情報 --------\n\
{merged_df.describe()}\n\
-------- 過去のデータに基づいた予測 --------\n\
Y=4, Z=1 の場合の p の事後平均: {posterior_mean:.4f}\n\
Y=4, Z=1 の場合の p の94% HDI区間: {hdi_interval}\n\
----------------------------------------\n\
キャパ38で、4ステ料金制の公演とする。\n\
1ステあたりの予測動員数: {38*posterior_mean}\n\
予測総動員数: {152*posterior_mean}\n\
1ステあたりの動員数の94%予測区間: {38*hdi_interval}\n\
総動員数の94%予測区間: {152*hdi_interval}"
with open("outputs/pymc_summary.txt", "w") as f:
    f.write(text)