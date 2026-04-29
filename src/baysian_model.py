import pymc as pm
import arviz as az
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
from pathlib import Path
from scipy.special import expit
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf # 追加

def load_and_preprocess_data(input_path: Path) -> pd.DataFrame:
    """データを読み込み、企画ごとに集計を行う"""
    df = pd.read_csv(input_path)
    
    # 合計に加えて、平均も計算するように修正
    grouped_df = df.groupby("企画", sort=False).agg(
        合計キャパ=("キャパ", "sum"),
        合計予約数=("予約数", "sum"),
        平均キャパ=("キャパ", "mean"),
        平均予約数=("予約数", "mean"),
        公演数=("企画", "count")
    )

    # 予約率（%）を計算
    grouped_df['予約率'] = (grouped_df['合計予約数'] / grouped_df['合計キャパ']) * 100
    
    print("----- データ -----")
    print(grouped_df.head())
    print("-----------------")
    print(grouped_df.describe())
    print("-----------------")
    print(f"データ数：{len(grouped_df)}")
    print("-----------------")
    
    return grouped_df

def save_histogram(df: pd.DataFrame, output_path: Path) -> None:
    """データフレームのヒストグラムを保存する"""
    numeric_df = df.select_dtypes(include=[np.number])
    ax = numeric_df.hist(figsize=(12, 8))
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def save_timeseries_plots(df: pd.DataFrame, output_path: Path) -> None:
    """企画ごとの平均予約数・平均キャパ数・予約率の時系列推移をプロットする"""
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    # 第1軸（左軸）：人数のプロット
    ax1.plot(df.index, df["平均キャパ"], marker='o', label="平均キャパ数", linestyle='-', color='tab:blue')
    ax1.plot(df.index, df["平均予約数"], marker='s', label="平均予約数", linestyle='--', color='tab:orange')
    ax1.set_xlabel("企画 (時系列順)")
    ax1.set_ylabel("人数", color='black')
    ax1.grid(True, alpha=0.5)

    # X軸の企画名が重ならないように45度回転させる
    ax1.tick_params(axis='x', rotation=45)
    
    # 第2軸（右軸）：割合のプロットを作成
    ax2 = ax1.twinx()
    ax2.plot(df.index, df["予約率"], marker='o', label="予約率", linestyle='-.', color='tab:red')
    ax2.set_ylabel("予約率 (割合)")
    
    # 左右の軸の凡例を1つにまとめる
    handler1, label1 = ax1.get_legend_handles_labels()
    handler2, label2 = ax2.get_legend_handles_labels()
    ax1.legend(handler1 + handler2, label1 + label2, loc="upper left")
    
    plt.title("企画ごとの平均キャパ数・平均予約数・予約率の推移")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def save_acf_pacf_plots(df: pd.DataFrame, output_path: Path) -> None:
    """平均予約数と予約率のACF, PACFをプロットする"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 平均キャパ数のACF / PACF
    plot_acf(df["予約率"].dropna(), ax=axes[0, 0], title="予約率の自己相関 (ACF)")
    # データ数が少ない場合、PACFの計算lag数を制限する（データ数の半分以下など）
    lags = min(10, len(df["予約率"]) // 2 - 1) 
    if lags > 0:
        plot_pacf(df["予約率"].dropna(), ax=axes[0, 1], lags=lags, title="予約率の偏自己相関 (PACF)", method='ywm')
    else:
        axes[0, 1].set_title("予約率のPACF (データ不足で計算不可)")

    # 合計キャパ数のACF / PACF
    plot_acf(df["平均予約数"].dropna(), ax=axes[1, 0], title="平均予約数の自己相関 (ACF)")
    if lags > 0:
        plot_pacf(df["平均予約数"].dropna(), ax=axes[1, 1], lags=lags, title="平均予約数の偏自己相関 (PACF)", method='ywm')
    else:
        axes[1, 1].set_title("平均予約数のPACF (データ不足で計算不可)")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def run_mcmc_model(df: pd.DataFrame) -> az.InferenceData:
    """MCMCサンプリングを実行し、結果を返す"""
    print("----- MCMCを実行 -----")
    
    N = df['合計キャパ'].values
    X = df['合計予約数'].values
    Y = df['公演数'].values
    mu = 0
    sigma = 10000

    with pm.Model() as logistic_model:

        # 事前分布
        beta_0 = pm.Normal('beta_0', mu=mu, sigma=sigma)
        beta_1 = pm.Normal('beta_1', mu=mu, sigma=sigma)

        # ロジットリンク関数
        eta = beta_0 + beta_1 * Y
        p = pm.math.invlogit(eta)

        # 尤度関数 (二項分布)
        pm.Binomial('X_obs', n=N, p=p, observed=X)

        # MCMCサンプリング
        trace = pm.sample(5000, tune=1000, cores=2, random_seed=42)

    print("### 1. パラメータ(beta)の事後分布の要約 ###")
    print(az.summary(trace, var_names=['beta_0', 'beta_1']))
    
    return trace

def predict_and_report(df: pd.DataFrame, trace: az.InferenceData, num_stages: int, capacity: int, report_path: Path) -> None:
    """予測を行い、結果をコンソールとファイルに出力する"""
    beta_0_samples = trace.posterior['beta_0'].values.flatten()
    beta_1_samples = trace.posterior['beta_1'].values.flatten()

    eta_new_samples = beta_0_samples + beta_1_samples * num_stages
    p_new_samples = expit(eta_new_samples) 

    posterior_mean = np.mean(p_new_samples)
    hdi_interval = az.hdi(p_new_samples, hdi_prob=0.94)

    print(f"num_stages：{num_stages}の p の事後平均: {posterior_mean:.4f}")
    print(f"num_stages：{num_stages}の p の94% HDI区間: {hdi_interval}")
    print("--------------------")
    print(f"キャパ{capacity}で、{num_stages}ステの公演とする。")
    print(f"1ステあたりの予測動員数: {capacity * posterior_mean:.1f}")
    print(f"予測総動員数: {capacity * num_stages * posterior_mean:.1f}")
    print(f"1ステあたりの動員数の94%予測区間: {capacity * hdi_interval}")
    print(f"総動員数の94%予測区間: {capacity * num_stages * hdi_interval}")
    print("--------------------")

    report_text = f"""-------- 過去の公演の基本的な情報 --------
{df.describe()}
-------- 過去のデータに基づいた予測 --------
num_stages：{num_stages}の p の事後平均: {posterior_mean:.4f}
num_stages：{num_stages}の p の94% HDI区間: {hdi_interval}
----------------------------------------
キャパ{capacity}で、{num_stages}ステの公演とする。
1ステあたりの予測動員数: {capacity * posterior_mean:.1f}
予測総動員数: {capacity * num_stages * posterior_mean:.1f}
1ステあたりの動員数の94%予測区間: {capacity * hdi_interval}
総動員数の94%予測区間: {capacity * num_stages * hdi_interval}
"""
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

def main():
    current_dir = Path.cwd()
    input_path = current_dir / "data" / "過去の本企画集客.csv"
    
    output_dir = current_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    histgram_path = output_dir / "histogram.png"
    timeseries_path = output_dir / "timeseries.png" # 追加
    acf_pacf_path = output_dir / "acf_pacf.png"     # 追加
    report_path = output_dir / "pymc_summary.txt"

    num_stages = 5     # ステ数
    capacity = 35      # キャパ

    try:
        df = load_and_preprocess_data(input_path)
        
        # グラフの保存処理
        save_histogram(df, histgram_path)
        save_timeseries_plots(df, timeseries_path) # 追加
        save_acf_pacf_plots(df, acf_pacf_path)     # 追加
        
        # モデルの実行と予測
        trace = run_mcmc_model(df)
        predict_and_report(df, trace, num_stages, capacity, report_path)
        
        print("処理が正常に完了し、グラフとレポートが outputs/ フォルダに出力されました。")
    except Exception as e:
        print(f"エラーが発生しました: {e}")

if __name__ == '__main__':
    main()