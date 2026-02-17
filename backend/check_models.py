import google.generativeai as genai
import os

# 環境変数にAPIキーが入っている前提
api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
    print("✖ GEMINI_API_KEY が環境変数に設定されていません")
    exit()
    
genai.configure(api_key=api_key)

print("=== embedContent が使えるモデル一覧 ===")

for m in genai.list_models():
    methods = getattr(m, "supported_generation_methods", [])
    if "embedContent" in methods:
        print(m.name, methods)