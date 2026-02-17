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
CHROMA_DB_PATH = r"C:\開発\社内用チャットボット\chat-bot-system\chroma_db"


# ========================
# 埋め込み取得関数
# ========================
def get_embedding(text: str, max_retries: int = 5, delay: float = 2.0) -> list[float]:
    if not text:
        return []
    
    for attempt in range(1, max_retries + 1):
        try:
            response = genai.embed_content(
                model="models/gemini-embedding-001",
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
        # ---- 新旧どちらの列名にも対応させる ----
        #（旧）日本語版:「質問」「回答」
        #（新）英語版：「question」「answer_internal」
        if '質問' in df.columns and '回答' in df.columns:
            question_col = '質問'
            answer_col = '回答'
        elif 'question' in df.columns and ('answer_internal' in df.columns or 'answer' in df.columns):  # ← colimns → columns に修正
            question_col = 'question'
            answer_col = 'answer_internal' if 'answer_internal' in df.columns else 'answer'
        else:
            logger.error("CSVファイルに質問・回答の列が見つかりません。")
            logger.error(f"検出された列: {list(df.columns)}")
            raise ValueError("Required columns ('質問', '回答') または ('question', 'answer_internal') が見つかりません。")

        df = df.dropna(subset=[question_col, answer_col])
        logger.info(f"使用カラム: 質問={question_col}, 回答={answer_col}")
        logger.info(f"読み込んだQ&Aデータ数: {len(df)} 件")
        # ✅ 修正ポイント：この関数内で定義したカラム名を返す
        return df, question_col, answer_col  # ← 修正！
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

    print("\n--- デバッグ情報：読み込んだqa_data.csvの内容 ---")
    try:
        qa_df = pd.read_csv(qa_file_path)
        print(qa_df.head(10)) # 先頭10行を表示
        print(f"\n合計件数: {len(qa_df)}")
        print("-------------------------------------------\n")
    except Exception as e:
        print(f"ERROR: qa_data.csvの読み込み中にエラーが発生しました: {e}")
        return
    
    # ✅ Gemini埋め込み関数を使って ChromaDB コレクションを作成
    embedding_func = GeminiEmbeddingFunction()
    
    client = PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_func
    )
    logger.info(f"ChromaDBコレクション '{collection_name}' を作成/取得しました。")
    
    # 既存データを削除して再構築
    logger.info("既存のChromaDBデータを削除します。")
    client.delete_collection(name=collection_name)
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_func
    )
    logger.info("ChromaDBコレクションを再作成しました。")
    
    try:
        # ✅ 修正ポイント：load_qa_data から3つの値を受け取る
        qa_df, question_col, answer_col = load_qa_data(qa_file_path)
        
        documents = []
        metadatas = []
        embeddings = []
        ids = []
        
        for index, row in qa_df.iterrows():
            question = str(row[question_col])
            answer = str(row[answer_col])
            
            # ★埋め込みは「質問」だけで作る（検索の当たりを良くする）
            doc_text = question

            embedding = get_embedding(doc_text)
            if not embedding:
                ...
            else:
                documents.append(doc_text)
                metadatas.append({
                    "question": question,
                    "answer": answer,
                    "source": qa_file_path,
                    "row_id": index,
                    # もしCSVに answer_id/intent/tags/audience があるなら併せて入れる
                    **({ "answer_id": str(row.get("answer_id")) } if "answer_id" in qa_df.columns else {}),
                    **({ "intent": str(row.get("intent")) } if "intent" in qa_df.columns else {}),
                    **({ "tags": str(row.get("tags")) } if "tags" in qa_df.columns else {}),
                    **({ "audience": str(row.get("audience")) } if "audience" in qa_df.columns else {}),
                })
                embeddings.append(embedding)
                ids.append(f"qa_{index}")
                
            time.sleep(1.0) # API制限対策
            
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
    
    # ★★★ 修正：script_dir を再定義 ★★★
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    qa_csv_path = os.path.join(script_dir, "qa_data.csv")
    logger.info(f"Q&A CSVパス: {qa_csv_path}")
    
    try:
        process_and_store_qa_data(qa_csv_path, collection_name="internal_qa_collection_v2")
        logger.info("Q&Aデータのベクトルデータベースへの保存が完了しました。")
    except Exception as e:
        logger.error(f"Q&Aデータ処理中にエラーが発生しました: {e}")
