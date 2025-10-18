# # tools/convert_to_internal.py
# import re
# import pandas as pd

# SRC = "qa_data.csv"              # 元CSV（質問, 回答）
# DST = "qa_data_internal.csv"     # 出力CSV（質問, 回答, 回答_社内向け）

# PLACEHOLDER = "（ここに詳しい内容を入れて）"

# # 電話誘導や問い合わせ誘導など、社内向けに不要な文言を置換
# PHONE_PATTERNS = [
#     r"詳しくご案内.*?(お電話|ご連絡).*",
#     r"お問い合わせ.*(フォーム|ページ).*",
#     r"(弊社|サポート|担当者).*?まで.*?ご連絡ください.*",
#     r"\(?0\d{1,4}\)?[-−ー‐ ]?\d{1,4}[-−ー‐ ]?\d{3,4}",  # 一般的な電話番号
#     r"0120[-−ー‐ ]?\d{3}[-−ー‐ ]?\d{3,4}",
#     r"詳しくは.*?をご確認ください。"
# ]
# VAGUE_PATTERNS = [
#     r"場合がございます。", r"必要に応じて.*?してください。", r"ご検討ください。"
# ]

# def to_internal(answer: str) -> str:
#     if not isinstance(answer, str) or not answer.strip():
#         return ""

#     a = answer

#     # 1) 電話・問い合わせ誘導をプレースホルダへ
#     for p in PHONE_PATTERNS:
#         a = re.sub(p, PLACEHOLDER, a, flags=re.IGNORECASE)

#     # 2) ふわっと表現をプレースホルダへ
#     for p in VAGUE_PATTERNS:
#         a = re.sub(p, PLACEHOLDER, a)

#     # ★テンプレは付けない。元の文章を必要最小限の置換だけで返す
#     return a

# def main():
#     df = pd.read_csv(SRC)
#     if not {'質問','回答'}.issubset(df.columns):
#         raise ValueError("CSVに '質問' と '回答' 列が必要です。")
#     df['回答_社内向け'] = df['回答'].apply(to_internal)
#     df.to_csv(DST, index=False, encoding="utf-8-sig")
#     print(f"変換完了: {DST}（質問, 回答, 回答_社内向け）")

# if __name__ == "__main__":
#     main()
