import os
import re
import time
import argparse
import datetime
import pandas as pd
import requests
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# ===== 設定 =====
API_URL = "http://localhost:8000/chat"  # backend/main.py を起動しておく
QA_CSV_PATH = os.path.join(os.path.dirname(__file__), "qa_data.csv")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

# 軽量エンコーダ（比較用）
encoder = SentenceTransformer("paraphrase-MiniLM-L6-v2")

# “見つかりません”系の返答を検出するパターン
FALLBACK_PATTERNS = [
    "見つかりません", "関連情報が見つかりません","見当たりませんでした",
    "わかりません", "分かりません", "わかりかねます", "わかり兼ねます",
    "他のキーワードでお試しください", "internal server error",
    "sorry", "i don't know", "no relevant information"
]

def is_not_found(text: str) -> bool:
    """“見つかりません”系の返答を検出（空文字も不正解扱い）"""
    if not text:
        return True
    t = re.sub(r"\s+", "", text.lower())  # 空白を除去して小文字化
    for pat in FALLBACK_PATTERNS:
        if pat.lower().replace(" ", "") in t:
            return True
    return False

def semantic_score(text1: str, text2: str) -> float:
    """文章同士の意味類似度（0.0〜1.0）"""
    if not text1 or not text2:
        return 0.0
    emb1 = encoder.encode([text1], convert_to_tensor=True)
    emb2 = encoder.encode([text2], convert_to_tensor=True)
    sim = cosine_similarity(emb1.cpu().numpy(), emb2.cpu().numpy())[0][0]
    return round(float(sim), 3)

def main(limit: int, threshold: float, sleep_sec: float):
    # ===== データ読み込み =====
    qa_df = pd.read_csv(QA_CSV_PATH)
    if "質問" not in qa_df.columns or "回答" not in qa_df.columns:
        raise ValueError("CSVに「質問」「回答」列が必要です")
    qa_df = qa_df.dropna(subset=["質問", "回答"])
    if limit:
        qa_df = qa_df.head(limit)

    print(f"📄 Q&Aデータ {len(qa_df)} 件をテストします（閾値 {threshold}）")

    # ===== 出力先準備（履歴 + 最新）=====
    os.makedirs(RESULTS_DIR, exist_ok=True)
    date_dir = os.path.join(RESULTS_DIR, datetime.datetime.now().strftime("%Y%m%d"))
    os.makedirs(date_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path_ts = os.path.join(date_dir, f"test_results_{timestamp}.csv")
    out_path_latest = os.path.join(RESULTS_DIR, "test_results_latest.csv")

    results = []
    correct = 0

    start_time = time.time()  # テスト開始時刻を記録

    for idx, row in qa_df.iterrows():
        q = str(row["質問"]).strip()
        expected = str(row["回答"]).strip()
        print(f"\n🟢 質問 {idx}: {q}")

        # ===== API呼び出し =====
        try:
            r = requests.post(API_URL, json={"message": q, "user_id": "eval"})
            r.raise_for_status()
            actual = r.json().get("response", "").strip()
        except Exception as e:
            print(f"❌ API呼び出し失敗: {e}")
            actual = f"[APIエラー] {e}"

        # ===== スコア計算（“見つかりません”系は強制 0.0）=====
        if is_not_found(actual):
            score = 0.0
        else:
            score = semantic_score(expected, actual)

        judge = "〇" if score >= threshold else "×"
        if judge == "〇":
            correct += 1

        print(f"期待回答: {expected[:60]}...")
        print(f"実際回答: {actual[:60]}...")
        print(f"スコア: {score} 判定: {judge}")

        results.append({
            "No": idx,
            "質問": q,
            "期待回答": expected,
            "実際回答": actual,
            "スコア": score,
            "判定": judge
        })

        time.sleep(sleep_sec)  # API制限対策

    # ===== 結果保存（履歴 + 最新）=====
    df = pd.DataFrame(results)
    df.to_csv(out_path_ts, index=False, encoding="utf-8-sig")
    df.to_csv(out_path_latest, index=False, encoding="utf-8-sig")

    # 正答率計算（パーセント表示）
    acc = round(correct / len(df) * 100, 1) if len(df) else 0.0
    print(f"\n✅ 完了: {len(df)}件 / 正解 {correct}件 / 正答率 {acc}%")
    
    # 経過時間表示
    end_time = time.time()  # テスト終了時刻を記録
    elapsed_time = end_time - start_time  # 経過時間（秒）
    
    # 経過時間を分・秒・時間の組み合わせで表示
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    hours = int(minutes // 60)
    minutes = minutes % 60

    elapsed_time_str = ""
    if hours > 0:
        elapsed_time_str += f"{hours}h"
    if minutes > 0:
        elapsed_time_str += f"{minutes}m"
    if seconds > 0:
        elapsed_time_str += f"{seconds}s"

    print(f"⏳ 時間: {elapsed_time_str}")  # 経過時間を表示

    print(f"🗂 履歴ファイル: {out_path_ts}")
    print(f"📌 最新ファイル: {out_path_latest}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=10, help="テスト件数（先頭から）")
    parser.add_argument("--threshold", type=float, default=0.6, help="正解判定のスコア閾値")
    parser.add_argument("--sleep", type=float, default=0.5, help="API呼び出しのインターバル秒")
    args = parser.parse_args()
    main(args.limit, args.threshold, args.sleep)
