import google.generativeai as genai
import os

# 事前に環境変数 GEMINI_API_KEY を設定していることが前提
# 例: setx GEMINI_API_KEY "xxxx"  (Windows PowerShell)

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# モデル一覧を取得
models = genai.list_models()

# 出力
print("=== 利用可能なモデル一覧 ===")
for m in models:
    print(m.name)
