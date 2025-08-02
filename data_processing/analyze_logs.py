import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import os
import re
# import mecab #日本語の単語分割ライブラリ（オプション）

#ログデータベースのパス
#analyza_logs.pyはdata_processingディレクトリにあるため、プロジェクトルートのdbファイルを指す
LOG_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),"..", "db", "chatbot_logs_db")

def get_query_logs():
    """
    データベースから問合せログを取得します。
    """
    conn = None
    try:
        conn = sqlite3.connect(LOG_DB_PATH)
        df = pd.read_sql_query("SELECT * FROM queries ORDER BY timestamp DESC", conn)
        return df
    except sqlite3.Error as e:
        print(f"Error reading log database: {e}")
        return pd.DataFrame() # 空のDataFrameを返す
    finally:
        if conn:
            conn.close()
def analyze_queries(df: pd.DataFrame):
    """
    問合せテキストを分析し、抽出キーワードなどを特定する。
    """
    if df.empty:
        print("ログデータがありません。")
        return
    
    print("\n--- 最新の問合せトップ5 ---")
    print(df[['timestamp', 'user_id', 'query_text', 'bot_response_text']].head())

    print("\n--- ユーザーごとの問合せ数 ---")
    user_counts = df['user_id'].value_counts()
    print(user_counts)

    #日本語の単語分割はMeCabなどのライブラリが必要になる。
    #環境構築が複雑なため、ここでは簡単な単語抽出のみを行うか、MeCab導入を別途検討
    #現時点では簡単なスペース区切りでの単語抽出に留める。
    print("\n--- 頻出キーワード（質問テキストから - 簡易版）---")
    all_words = []
    for query in df['query_text']:
        #全角スペース、半角スペース、句読点などで分割し、小文字化
        words = re.findall(r'\b\w+\b', query.lower())
        all_words.extend(words)

    #頻出する上位10位
    #あまりにも一般的な単語(助詞)などは除外するフィルタリングを追加すると良い
    stop_words = ["は", "が", "を", "に", "で", "と", "も", "です", "ます", "ください", "の", "これ", "それ", "あの", "質問", "教えて" ]
    filtered_words = [word for word in all_words if word not in stop_words and len(word) > 1] #1文字の単語も除外

    most_common_words = Counter(filtered_words).most_common(10)
    if most_common_words:
        for word, count in most_common_words:
            print(f"- {word}: {count}回")

        #オプション：棒グラフで可視化
        words, counts = zip(*most_common_words)
        plt.figure(figsize=(10, 6))
        plt.bar(words, counts, color='skyblue')
        plt.xlabel('キーワード')
        plt.ylabel('出現回数')
        plt.title('頻出キーワードトップ10')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
    else:
        print("分析できるキーワードが見つかりませんでした。")

if __name__ == "__main__":
    logs_df = get_query_logs()
    analyze_queries(logs_df)
        