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
    LOG_DB_PATH = os.path.join(LOG_DB_DIR, "chatbot_logs.db")
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
                       bot response_text TEXT NOT NULL,
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
    ユーザーの問合せ、ボットの応答、使用したコンテキストをデータベースに保存する。
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
    allow_origins=["*"], #★本番環境では具体的なオリジンに修正する★
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Gemini APIキー読み込み
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.error("Gemini APIキーが設定されていません。")

# Gemini API設定 (APIキーがある場合のみ)
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    logger.warning("GEMINI_API_KEYが設定されていないため、Gemini APIへの呼び出しは失敗します。")

# ========================
# ChromaDB 設定
# ========================
CHROMA_DB_PATH = "/tmp/chroma_db"
COLLECTION_NAME = "internal_qa_collection_v2" 

# ========================
# GCSバケット名とパスの定義
# ========================
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "gemini-chatbot-project-v2-chroma-db-data") 
GCS_DB_PREFIX = os.getenv("GCS_DB_PREFIX", "chroma_db/")

if os.environ.get("K_SERVICE"):
    logger.info("Cloud Run環境を検出しました。ChromaDBデータをGCSからダウンロードします。")
    try:
        client = storage.Client()
        bucket = client.bucket(GCS_BUCKET_NAME)

        os.makedirs(CHROMA_DB_PATH, exist_ok=True)
        logger.info(f"ChromaDBダウンロード先ディレクトリ: {CHROMA_DB_PATH} を作成しました。")
        
        # 環境変数から読み込んだ直後に追加
        logger.info(f"GCS_BUCKET_NAME (env): {GCS_BUCKET_NAME}")
        logger.info(f"GCS_DB_PREFIX  (env): {GCS_DB_PREFIX}")

        blobs_found = False
        for blob in bucket.list_blobs(prefix=GCS_DB_PREFIX):
            if blob.name.endswith('/'):
                continue

            blobs_found = True
            local_rel_path = os.path.relpath(blob.name, GCS_DB_PREFIX)
            local_path = os.path.join(CHROMA_DB_PATH, local_rel_path)

            if os.path.dirname(local_path) and not os.path.exists(os.path.dirname(local_path)):
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                logger.info(f"ローカルディレクトリ: {os.path.dirname(local_path)} を作成しました。")

            logger.info(f"ダウンロード中: gs://{GCS_BUCKET_NAME}/{blob.name} -> {local_path}")
            blob.download_to_filename(local_path)
            logger.info(f"ダウンロード完了: {blob.name}")
            
        if not blobs_found:
            logger.warning(f"GCSバケット '{GCS_BUCKET_NAME}' のプレフィックス '{GCS_DB_PREFIX}' にファイルが見つかりませんでした。ChromaDBデータが正しくアップロードされているか確認してください。")
            raise FileNotFoundError(f"No ChromaDB files found in GCS bucket '{GCS_BUCKET_NAME}' with prefix '{GCS_DB_PREFIX}'")
        
        logger.info("ChromaDBデータのダウンロードが完了しました。")
    
    except Exception as e:
        logger.error(f"GCSからのChromaDBデータダウンロード中に致命的なエラーが発生しました: {e}", exc_info=True)
        raise

else:
    local_chroma_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "chroma_db")
    if not os.path.exists(local_chroma_path):
        logger.error(f"ローカル環境でChromaDBディレクトリが見つかりません: {local_chroma_path}")
        raise FileNotFoundError(f"Local ChromaDB directory not found: {local_chroma_path}")
    
    CHROMA_DB_PATH = local_chroma_path
    logger.info("ローカル環境を検出しました。ローカルのChromaDBデータを使用します。")

# ========================
# get_embedding 関数
# ========================
def get_embedding(text: str, task_type: str, max_retries: int = 5, delay: float = 2.0) -> list[float]:
    if not text:
        return []
    
    for attempt in range(1, max_retries + 1):
        try:
            if not GEMINI_API_KEY:
                logger.error("Gemini APIキーが設定されていないため、埋め込み生成をスキップします。")
                return []
            
            response = genai.embed_content(
                model="models/embedding-001",
                content=text,
                task_type=task_type
             )
            embedding = response.get("embedding")
            if not embedding:
                embedding = response.get("data", [{}])[0].get("embedding")
                if not embedding:
                    logger.error("埋め込みが取得できませんでした。レスポンス: %s", response)
                    return []
            logger.info(f"埋め込みの長さ: {len(embedding)}")
            return embedding
        
        except Exception as e:
            error_message = str(e)
            if "429" in error_message or "quota" in error_message.lower():
                logger.warning(f"[{attempt}/{max_retries}] 429エラーまたはクォータ制限。{delay}秒後にリトライします。")
                time.sleep(delay)
                delay *= 2
            else:
                logger.exception(f"Gemini API埋め込み取得中にエラーが発生しました。テキスト: '{text[:50]}...'")
                raise
            
    logger.error("最大リトライ回数に達しました。埋め込み取得を中断します。")
    return []
                
# ========================
# GeminiEmbeddingFunctionクラス
# ========================
class GeminiEmbeddingFunction(DefaultEmbeddingFunction):
    def __call__(self, texts):
        # ChromaDBのドキュメント埋め込みにはretrieval_documentを使用
        return [get_embedding(text, task_type="retrieval_document") for text in texts]

# embedding_func を GeminiEmbeddingFunction に差し替え
embedding_func = GeminiEmbeddingFunction()

# ★修正: ChromaDBのクライアントとコレクションの初期化をここで行う★
db_client = PersistentClient(path=CHROMA_DB_PATH)
try:
    qa_collection = db_client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_func
    )
    logger.info(f"ChromaDBコレクション '{COLLECTION_NAME}' を初期化しました。アイテム数: {qa_collection.count()}")
except Exception as e:
    logger.error(f"ChromaDBコレクションの初期化に失敗しました: {e}", exc_info=True)
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
    
    bot_response = ""
    context = ""
    
    try:
        if not GEMINI_API_KEY:
            bot_response = "申し訳ありませんが、Gemini APIキーが設定されていないため、現在AIによる応答ができません。"
            logger.warning("Gemini APIキーが設定されていないため、代替応答を返します。")
            return {"response": bot_response}
        
        # Geminiモデルで質問を拡張する
        query_expansion_model = genai.GenerativeModel(model_name="models/gemini-2.5-flash")
        query_expansion_prompt = f"""
以下のユーザーの質問に対して、ChromaDBから関連情報を検索するために最適化された、3〜5個の異なる検索クエリを生成してください。
質問の意図を正確に捉え、具体的なキーワードを含めるようにしてください。
回答は、各クエリを改行で区切ったリスト形式にしてください。

### ユーザーの質問
{request.message}

### 生成する検索クエリの例:
- 休暇申請の方法
- 有給休暇の取得手順
- 休暇の承認プロセス
- 休暇申請書の提出先

### あなたの返答（検索クエリのリスト形式）
"""
        expansion_response = query_expansion_model.generate_content(query_expansion_prompt)
        expanded_queries = expansion_response.text.strip().split('\n')
        
        expanded_queries = expanded_queries[:5]

        logger.info(f"生成された検索クエリ:\n{expanded_queries}")
        
        # 複数のクエリでChromaDBからベクトル検索
        all_answers = []
        for query in expanded_queries:
            query_embedding = get_embedding(query, task_type="retrieval_query")
            if not query_embedding:
                continue

            results = qa_collection.query(
                query_embeddings=[query_embedding],
                n_results=10,
                include=["metadatas"]  # ← documentsではなくmetadatasだけ
            )
            if results and results.get("metadatas"):
                for metadata in results["metadatas"][0]:
                    q = metadata.get("question")
                    a = metadata.get("answer")
                    if q and a:
                        all_answers.append(f"質問: {q}\n回答: {a}")

        # 重複削除＋最大5件に制御
        unique_answers = list(dict.fromkeys(all_answers))[:5]
        context = "\n\n".join(f"- {pair}" for pair in unique_answers)
        logger.info(f"検索結果（質問+回答ペア）:\n{context}")

        
        if not context.strip():
            logger.info("ChromaDBに関連情報は見つかりませんでした。")
            bot_response = "申し訳ありませんが、その質問に対する関連情報が見つかりませんでした。他のキーワードでお試しください。"
        else:
            logger.info(f"検索結果（回答文）:\n{context}")
            
            # Geminiプロンプト作成（要約＋自然な回答生成）
            final_answer_model = genai.GenerativeModel(model_name="models/gemini-2.5-pro")
            prompt = f"""
あなたは社内のサポート用チャットボットです。
ユーザーの質問に対して、以下の【参考情報】を基に、簡潔かつ丁寧に回答してください。

- 【参考情報】に回答がない場合は、「申し訳ありませんが、その質問に対する関連情報が見つかりませんでした。他のキーワードでお試しください。
- 質問に回答する際、コンテキストにある情報以外のことを付け足してはいけません。
- 丁寧な言葉遣いで回答してください。

### ユーザーの質問
{request.message}

### 参考情報
{context}

### あなたの返答（丁寧に自然な言い回しで）
"""
            response = final_answer_model.generate_content(prompt)
            bot_response = response.text.strip()
            
            logger.info(f"Gemini response: {bot_response}")
            
    except Exception as e:
        logger.exception("Geminiチャット処理でエラーが発生しました。")
        bot_response = f"An internal server error occurred: {e}"
        
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {e}")
    
    finally:
        save_query_log(
            user_id=request.user_id,
            query_text=request.message,
            bot_response_text=bot_response,
            context_used=context
        )
    
    return {"response": bot_response}

# Uvicorn起動（開発用）
if __name__ == "__main__":
    init_log_db() 
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)