import os
import logging
import sqlite3
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
from chromadb import PersistentClient
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
import time
from google.cloud import storage
import pandas as pd

# ロギング設定
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

#=========================
#ログデータベース設定
#=========================
#環境によってログDBのパスを切り替える
if os.environ.get("K_SERVICE"):
    # Cloud Run環境の場合
    LOG_DB_DIR = os.path.join("/tmp", "db")
else:
    #ローカル環境の場合、プロジェクトルートのdbディレクトリに配置
    LOG_DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "db")
LOG_DB_PATH = os.path.join(LOG_DB_DIR, "chatbot_logs.db")

#=========================
#関数の定義(init_log_dbとsave_query_logをここに移動)
#=========================

#=========================
# init_log_db 関数
#=========================
def init_log_db():
    """
    問合せログを保存するためのSQLiteデータベースとテーブルを初期化する。
    """
    os.makedirs(LOG_DB_DIR, exist_ok=True)
    logger.info(f"Log database directory created at {LOG_DB_DIR}")

    conn = None
    try:
        conn = sqlite3.connect(LOG_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS queries(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                query_text TEXT NOT NULL,
                bot_response_text TEXT NOT NULL,
                context_used TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        logger.info(f"Log database initialized at {LOG_DB_PATH}")
    except sqlite3.Error as e:
        logger.error(f"Error initializing log database: {e}", exc_info=True)
        raise
    finally:
        if conn:
            conn.close()

#=========================
# save_query_log 関数
#=========================
def save_query_log(user_id: str, query_text: str, bot_response_text: str, context_used: str = None):
    """
    ユーザーの問い合せ、ボットの応答、使用したコンテキストをデータベースに保存する。
    """
    conn = None
    try:
        conn = sqlite3.connect(LOG_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO queries (user_id, query_text, bot_response_text, context_used)
            VALUES (?, ?, ?, ?)
        """, (user_id, query_text, bot_response_text, context_used))
        conn.commit()
        logger.info(f"Query logged for user '{user_id}'.")
    except sqlite3.Error as e:
        logger.error(f"Error saving query log: {e}", exc_info=True)
    finally:
        if conn:
            conn.close()

# FastAPI アプリ
app = FastAPI()

#データベース初期化関数の呼び出し
init_log_db() 

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # ★本番環境では具体的なオリジンに修正してください★
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Gemini APIキー読み込み
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.error("Gemini APIキーが設定されていません。")
    # raise RuntimeError("Gemini APIキーが設定されていません。") # Cloud Runでの起動失敗を防ぐため、ここではアプリは起動させる
    # Cloud Runデプロイ時に環境変数で設定するため、このエラーで停止しないようにします。
    # ただし、APIキーが設定されていない場合、GenAIの呼び出しでエラーになります。

# Gemini API設定 (APIキーがある場合のみ)
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    logger.warning("GEMINI_API_KEYが設定されていないため、Gemini APIへの呼び出しは失敗します。")


# ========================
# ChromaDB 設定
# ========================
# Dockerコンテナ内では /tmp が推奨される一時ディレクトリ
# ★修正: Dockerコンテナ内でのパスを/tmp/chroma_dbに固定★
CHROMA_DB_PATH = "/tmp/chroma_db"
COLLECTION_NAME = "internal_qa_collection_v2" # コレクション名はこれでOK

# ========================
# GCSバケット名とパスの定義
# ========================
# ★修正: GCS_BUCKET_NAME をあなたのバケット名に置き換える (例: "gemini-chatbot-project-465409-chroma-db-data")★
GCS_BUCKET_NAME = "gemini-chatbot-project-465409-chroma-db-data" # ★★★ここをあなたのGCSバケット名に置き換えてください★★★
GCS_DB_PREFIX = "chroma_db/" # GCS上のディレクトリ名。末尾にスラッシュを付けた方が良い

# Cloud Run環境でのみChromaDBデータをGCSからダウンロード
if os.environ.get("K_SERVICE"): # K_SERVICE 環境変数はCloud Runで設定される
    logger.info("Cloud Run環境を検出しました。ChromaDBデータをGCSからダウンロードします。")
    try:
        client = storage.Client()
        bucket = client.bucket(GCS_BUCKET_NAME)

        # ChromaDBディレクトリが存在しない場合は作成
        os.makedirs(CHROMA_DB_PATH, exist_ok=True)
        logger.info(f"ChromaDBダウンロード先ディレクトリ: {CHROMA_DB_PATH} を作成しました。")

        blobs_found = False
        # GCSからファイルをリストアップし、ダウンロード
        for blob in bucket.list_blobs(prefix=GCS_DB_PREFIX):
            # フォルダ自体はダウンロードしない (空のオブジェクトの場合があるため)
            if blob.name.endswith('/'):
                continue
            
            blobs_found = True
            # GCS上のパスからローカルの相対パスを決定
            # 例: chroma_db/segment/ -> segment/
            local_rel_path = os.path.relpath(blob.name, GCS_DB_PREFIX)
            local_path = os.path.join(CHROMA_DB_PATH, local_rel_path)

            # ディレクトリが存在しない場合は作成
            # os.path.dirname(local_path) はファイルの親ディレクトリパスを返す
            if os.path.dirname(local_path) and not os.path.exists(os.path.dirname(local_path)):
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                logger.info(f"ローカルディレクトリ: {os.path.dirname(local_path)} を作成しました。")

            logger.info(f"ダウンロード中: gs://{GCS_BUCKET_NAME}/{blob.name} -> {local_path}")
            blob.download_to_filename(local_path)
            logger.info(f"ダウンロード完了: {blob.name}")

        if not blobs_found:
            logger.warning(f"GCSバケット '{GCS_BUCKET_NAME}' のプレフィックス '{GCS_DB_PREFIX}' にファイルが見つかりませんでした。ChromaDBデータが正しくアップロードされているか確認してください。")
            # ファイルが見つからない場合は、ChromaDBの初期化でエラーになる可能性が高いため、ここで例外を発生させる
            raise FileNotFoundError(f"No ChromaDB files found in GCS bucket '{GCS_BUCKET_NAME}' with prefix '{GCS_DB_PREFIX}'")

        logger.info("ChromaDBデータのダウンロードが完了しました。")

    except Exception as e:
        logger.error(f"GCSからのChromaDBデータダウンロード中に致命的なエラーが発生しました: {e}", exc_info=True)
        # エラー発生時はアプリケーションを起動させない方が良いので、ここでは例外を再発生させます
        raise # ★エラー発生時は例外を再発生させてコンテナを停止させる★
else:
    # ローカル環境の場合、chroma_dbディレクトリがbackendの親ディレクトリにあることを想定
    local_chroma_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "chroma_db")
    if not os.path.exists(local_chroma_path):
        logger.error(f"ローカル環境でChromaDBディレクトリが見つかりません: {local_chroma_path}")
        raise FileNotFoundError(f"Local ChromaDB directory not found: {local_chroma_path}") # ローカルテスト時に起動停止させる

    CHROMA_DB_PATH = local_chroma_path
    logger.info("ローカル環境を検出しました。ローカルのChromaDBデータを使用します。")

# ========================
# get_embedding 関数（process_qa.pyからコピー）
# ========================
def get_embedding(text: str, max_retries: int = 5, delay: float = 2.0) -> list[float]:
    if not text:
        return []

    for attempt in range(1, max_retries + 1):
        try:
            # Gemini APIキーが設定されていない場合はスキップ
            if not GEMINI_API_KEY:
                logger.error("Gemini APIキーが設定されていないため、埋め込み生成をスキップします。")
                return []

            response = genai.embed_content(
                model="models/embedding-001",
                content=text,
                task_type="retrieval_document"
            )
            embedding = response.get("embedding")
            if not embedding:
                embedding = response.get("data", [{}])[0].get("embedding")
                if not embedding:
                    logger.error("埋め込みが取得できませんでした。レスポンス: %s", response)
                    return []
            logger.info(f"main.pyで生成された埋め込みの長さ: {len(embedding)}")
            return embedding

        except Exception as e:
            error_message = str(e)
            if "429" in error_message or "quota" in error_message.lower():
                logger.warning(f"[{attempt}/{max_retries}] 429エラーまたはクォータ制限。{delay}秒後にリトライします。")
                time.sleep(delay)
                delay *= 2
            else:
                logger.exception(f"Gemini API埋め込み取得中にエラーが発生しました。テキスト: '{text[:50]}...'")
                raise # 埋め込み生成に失敗したら例外を発生させる


    logger.error("最大リトライ回数に達しました。埋め込み取得を中断します。")
    return []

# ========================
# GeminiEmbeddingFunctionクラス（process_qa.pyからコピー）
# ========================
class GeminiEmbeddingFunction(DefaultEmbeddingFunction):
    def __call__(self, texts):
        return [get_embedding(text) for text in texts]

# embedding_func を GeminiEmbeddingFunction に差し替え
embedding_func = GeminiEmbeddingFunction()

# ★修正: ChromaDBのクライアントとコレクションの初期化をここで行う★
# アプリケーション起動時に一度だけ実行されるように
db_client = PersistentClient(path=CHROMA_DB_PATH)
try:
    qa_collection = db_client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_func
    )
    logger.info(f"ChromaDBコレクション '{COLLECTION_NAME}' を初期化しました。アイテム数: {qa_collection.count()}")
except Exception as e:
    logger.error(f"ChromaDBコレクションの初期化に失敗しました: {e}", exc_info=True)
    # ChromaDBが起動できない場合はアプリケーションも起動できないため、ここで終了
    raise RuntimeError(f"Failed to initialize ChromaDB collection: {e}")


# リクエスト定義
class ChatRequest(BaseModel):
    message: str
    user_id: str = "anonymous"

@app.get("/")
async def read_root():
    return {"message": "Chatbot backend is running."}

@app.post("/chat")
async def chat_with_bot(request: ChatRequest):
    logger.info(f"[User: {request.user_id}] Message received: {request.message}")

    bot_response = "" # bot_response変数をtryブロックの外で初期化
    context = ""      # context変数をtryブロックの外で初期化
    
    try:
        # ChromaDBからベクトル検索
        query_embedding = get_embedding(request.message)
        if not query_embedding:
            raise ValueError("クエリの埋め込み生成に失敗しました。")
        
        # 明示的に生成した埋め込みを使ってクエリする
        results = qa_collection.query(
            query_embeddings=[query_embedding],
            n_results=3,
            include=["documents", "metadatas"]
        )

        # 回答文だけを抽出してまとめる（類似回答3件）
        if results and results.get("documents"):
            documents = results["documents"][0]
            context = "\n\n".join(f"- {doc}" for doc in documents)
            logger.info(f"検索結果（回答文）:\n{context}")
        else:
            context = ""
            logger.info("ChromaDBに関連情報は見つかりませんでした。")

        # Geminiプロンプト作成（要約＋自然な回答生成）
        if not GEMINI_API_KEY:
            bot_response = "申し訳ありませんが、Gemini APIキーが設定されていないため、現在AIによる応答ができません。"
            logger.warning("Gemini APIキーが設定されていないため、代替応答を返します。")
        else:
            model = genai.GenerativeModel(model_name="models/gemini-1.5-pro-latest")
            prompt = f"""
あなたは社内のサポート用チャットボットです。
以下は、過去のQ&Aの中から類似度が高かった回答例です。
これらを参考に、ユーザーの質問に対して自然な返答を作成してください。

### 類似回答候補（最大3件）
{context}

### ユーザーの質問
{request.message}

### あなたの返答（丁寧に自然な言い回しで）
"""
            response = model.generate_content(prompt)
            bot_response = response.text.strip()
            logger.info(f"Gemini response: {bot_response}")

    except Exception as e:
        logger.exception("Geminiチャット処理でエラーが発生しました。")
        bot_response = f"An internal server error occurred: {e}" # エラーメッセージをbot_responseに格納
        
        # ここに raise HTTPException を残します
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {e}")
    
    finally:
        # エラー発生時も、正常時も、必ずログを保存
        save_query_log(
            user_id=request.user_id,
            query_text=request.message,
            bot_response_text=bot_response,
            context_used=context # contextはtryブロック内で初期化されているため、locals()チェックは不要
        )
    
    # return は finally ブロックの直後に配置します
    return {"response": bot_response}

# Uvicorn起動（開発用）
if __name__ == "__main__":
    init_log_db() #データベース初期化関数の呼び出し
    
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
