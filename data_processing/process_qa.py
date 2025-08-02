import os
import time
import google.generativeai as genai
from chromadb import PersistentClient
import pandas as pd
import logging

# ✅ 追加: ChromaDBのEmbeddingFunctionクラスを使う
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

# ✅ 追加: Geminiの埋め込み関数をChromaDBと接続するためのラッパー関数
class GeminiEmbeddingFunction(DefaultEmbeddingFunction):
    def __call__(self, texts):
        return [get_embedding(text) for text in texts]

# ========================
# ログ設定
# ========================
SHOW_EMBED_LOG = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ========================
# APIキー設定
# ========================
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    logger.error("環境変数 GEMINI_API_KEY が設定されていません。")
    raise ValueError("環境変数 GEMINI_API_KEY が設定されていません。")
genai.configure(api_key=api_key)

# ========================
# ChromaDB保存先
# ========================
CHROMA_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "chroma_db")

# ========================
# 埋め込み取得関数
# ========================
def get_embedding(text: str, max_retries: int = 5, delay: float = 2.0) -> list[float]:
    if not text:
        return []

    for attempt in range(1, max_retries + 1):
        try:
            response = genai.embed_content(
                model="models/embedding-001",
                content=text,
                task_type="retrieval_document"
            )

            if SHOW_EMBED_LOG:
                logger.info(f"Geminiからの埋め込みレスポンス: {response}")

            embedding = response.get("embedding")
            if not embedding:
                embedding = response.get("data", [{}])[0].get("embedding")
                if not embedding:
                    logger.error("埋め込みが取得できませんでした。レスポンス: %s", response)
                    return []
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
# CSV読み込み関数
# ========================
def load_qa_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        if '質問' not in df.columns or '回答' not in df.columns:
            logger.error("CSVファイルに'質問'または'回答'の列が見つかりません。")
            raise ValueError("Required columns '質問' and '回答' not found in CSV.")
        df = df.dropna(subset=['質問', '回答'])

        logger.info(f"読み込んだQ&Aデータ数: {len(df)} 件")

        return df
    except FileNotFoundError:
        logger.error(f"ファイルが見つかりません: {file_path}")
        raise
    except Exception as e:
        logger.exception(f"Q&Aデータの読み込み中にエラーが発生しました: {e}")
        raise

# ========================
# メイン処理関数
# ========================
def process_and_store_qa_data(qa_file_path: str, collection_name: str = "internal_qa_collection"):
    logger.info(f"ChromaDBの保存先ディレクトリ: {CHROMA_DB_PATH}")
    os.makedirs(CHROMA_DB_PATH, exist_ok=True)

    # ✅ Gemini埋め込み関数を使って ChromaDB コレクションを作成
    embedding_func = GeminiEmbeddingFunction()

    client = PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_func
    )
    logger.info(f"ChromaDBコレクション '{collection_name}' を作成/取得しました。")

    try:
        qa_df = load_qa_data(qa_file_path)

        documents = []
        metadatas = []
        embeddings = []
        ids = []

        for index, row in qa_df.iterrows():
            question = str(row['質問'])
            answer = str(row['回答'])
            combined_text = f"質問: {question}\n回答: {answer}"

            embedding = get_embedding(combined_text)

            if not embedding:
                logger.warning(f"ベクトルが空です。質問: {question[:30]}... 回答: {answer[:30]}...")
                logger.warning(f"ID '{index}' のデータで埋め込み生成に失敗しました。スキップします。")
            else:
                logger.info(f"生成されたベクトルの長さ: {len(embedding)}")
                documents.append(answer)
                metadatas.append({"question": question, "source": qa_file_path, "row_id": index})
                embeddings.append(embedding)
                ids.append(f"qa_{index}")

            time.sleep(1.0)  # 1秒待機（API制限対策）

        if not documents:
            logger.warning("処理対象のQ&Aデータが見つかりませんでした。")
            return

        logger.info("collection.add を実行します。")
        start_time = time.time()

        collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

        elapsed = time.time() - start_time
        logger.info(f"collection.add 実行完了（所要時間: {elapsed:.2f}秒）")

        logger.info(f"ChromaDBに {len(documents)} 件のQ&Aデータを正常に保存しました。")
        logger.info("✅ collection.add 実行後の確認ログ：保存は最後まで到達しました。")

    except Exception as e:
        logger.exception("Q&Aデータの処理と保存中に予期せぬエラーが発生しました。")
        raise

# ========================
# スクリプトの実行
# ========================
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(script_dir, "..")
    CHROMA_DB_PATH = os.path.join(project_root, "chroma_db")

    qa_csv_path = os.path.join(script_dir, "qa_data.csv")
    logger.info(f"Q&A CSVパス: {qa_csv_path}")

    try:
        # ここでコレクション名を変更可能に
        process_and_store_qa_data(qa_csv_path, collection_name="internal_qa_collection_v2")
        logger.info("Q&Aデータのベクトルデータベースへの保存が完了しました。")
    except Exception as e:
        logger.error(f"Q&Aデータ処理中にエラーが発生しました: {e}")
