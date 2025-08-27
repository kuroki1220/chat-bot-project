# backend/tests/test_main.py

import pytest
import sqlite3
import os
import tempfile
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
import numpy as np
from pathlib import Path
from backend.main import (
    app,
    save_query_log,
    LOG_DB_PATH,
    init_log_db,
    ChatRequest,
    get_embedding,
    GeminiEmbeddingFunction
)

# FastAPIのテストクライアントを準備
client = TestClient(app)

# --- フィクスチャ ---
@pytest.fixture(scope="session", autouse=True)
def setup_log_db():
    """
    テストセッション全体で一度だけ実行されるフィクスチャ。
    テスト用のデータベースを初期化します。
    """
    # OSに依存しない一時的なディレクトリを安全に作成
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_db_path = Path(temp_dir) / "test_chatbot_logs.db"
        
        # LOG_DB_PATHを一時的に差し替える
        with patch('backend.main.LOG_DB_PATH', str(temp_db_path)):
            # init_log_dbを呼び出し、テーブルを初期化
            init_log_db()
            yield
            # `tempfile.TemporaryDirectory`のwith文を抜けると自動的に削除される

@pytest.fixture
def db_connection():
    """
    各テスト関数にデータベースコネクションを提供するフィクスチャ。
    """
    conn = sqlite3.connect(LOG_DB_PATH)
    yield conn
    conn.close()

# --- 単体テスト ---
def test_init_log_db_creates_table():
    """
    init_log_db関数がデータベースとテーブルを正しく作成するかテストする。
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_db_path = Path(temp_dir) / "test_init.db"
        
        with patch('backend.main.LOG_DB_PATH', str(temp_db_path)):
            with patch('backend.main.LOG_DB_DIR', temp_dir):
                init_log_db()
                assert os.path.exists(temp_db_path)
                
                conn = sqlite3.connect(temp_db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='queries';")
                result = cursor.fetchone()
                conn.close()
                assert result is not None
                assert result[0] == 'queries'

def test_save_query_log_success():
    """
    save_query_log関数がログを正しく保存するかテストする。
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_db_path = Path(temp_dir) / "test_chatbot_logs.db"
        with patch('backend.main.LOG_DB_PATH', str(temp_db_path)):
            init_log_db()
            user_id = "test_user_001"
            query = "テスト質問"
            response = "テスト回答"
            context = "テストコンテキスト"
            save_query_log(user_id, query, response, context)
            conn = sqlite3.connect(temp_db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT user_id, query_text, bot_response_text, context_used FROM queries WHERE user_id = ?", (user_id,))
            result = cursor.fetchone()
            conn.close()
            assert result is not None
            assert result[0] == user_id
            assert result[1] == query
            assert result[2] == response
            assert result[3] == context

@patch('backend.main.genai')
def test_get_embedding_success(mock_genai):
    """
    get_embedding関数が正常に埋め込みを返すかテストする。
    """
    mock_genai.embed_content.return_value = {"embedding": [0.1, 0.2, 0.3]}
    with patch('backend.main.GEMINI_API_KEY', 'fake_key'):
        embedding = get_embedding("テストテキスト")
        assert len(embedding) == 3
        assert embedding[0] == 0.1
        mock_genai.embed_content.assert_called_once_with(
            model="models/embedding-001",
            content="テストテキスト",
            task_type="retrieval_document"
        )

@patch('backend.main.genai')
def test_get_embedding_returns_empty_list_on_no_api_key(mock_genai):
    """
    APIキーが設定されていない場合、get_embeddingが空のリストを返すかテストする。
    """
    with patch('backend.main.GEMINI_API_KEY', None):
        embedding = get_embedding("テストテキスト")
        assert embedding == []
        mock_genai.embed_content.assert_not_called()

@patch('backend.main.get_embedding')
def test_gemini_embedding_function_calls_get_embedding(mock_get_embedding):
    """
    GeminiEmbeddingFunctionがget_embeddingを正しく呼び出すかテストする。
    """
    # モックの戻り値をNumPy配列に修正
    mock_get_embedding.return_value = np.array([0.1, 0.2, 0.3])
    
    embedding_func = GeminiEmbeddingFunction()
    texts = ["text1", "text2"]
    embeddings = embedding_func(texts)
    
    # 戻り値の形式と長さを検証
    assert len(embeddings) == 2
    # NumPy配列の比較には np.allclose() や assert np.array_equal() が推奨される
    # 今回は純粋なリストと比較するので、`==`で問題無いが、より安全なアサーションが望ましい
    assert np.array_equal(embeddings, [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]])
    
    # get_embeddingが2回呼び出されたか検証
    assert mock_get_embedding.call_count == 2
    
    # 呼び出し時の引数が正しかったか検証
    mock_get_embedding.assert_any_call("text1")
    mock_get_embedding.assert_any_call("text2")

@patch('backend.main.genai')
def test_get_embedding_no_embedding_field(mock_genai):
    """
    APIレスポンスにembeddingフィールドがない場合、空のリストを返すかテストする。
    """
    # 'embedding'フィールドを含まないレスポンスをモック
    mock_genai.embed_content.return_value = {"data": [{"embedding": None}], "metadata": {}}
    with patch('backend.main.GEMINI_API_KEY', 'fake_key'):
        embedding = get_embedding("テストテキスト")
        assert embedding == []
        mock_genai.embed_content.assert_called_once()

@patch('backend.main.sqlite3.connect')
def test_save_query_log_error_handling(mock_connect):
    """
    ログ保存に失敗した場合でも、関数が例外を処理するかテストする。
    """
    # 接続時に例外を発生させるようにモック
    mock_connect.side_effect = sqlite3.Error("Mock DB Error")

    user_id = "test_user_002"
    query = "テスト質問"
    response = "テスト回答"
    context = "テストコンテキスト"

    # 例外が発生しないことを確認
    try:
        save_query_log(user_id, query, response, context)
    except Exception as e:
        pytest.fail(f"ログ保存関数が予期せぬ例外を発生させました: {e}")

# --- 内部結合テスト ---
@patch('backend.main.genai')
@patch('backend.main.qa_collection')
def test_chat_endpoint_success(mock_qa_collection, mock_genai):
    """
    /chatエンドポイントがRAGプロセス全体を通して正しく動作するかをテストする。
    """
    with patch('backend.main.get_embedding', return_value=[0.1, 0.2, 0.3]):
        mock_qa_collection.query.return_value = {
            "documents": [["これはQ&Aデータから取得したコンテキストです。"]],
            "metadatas": [[{"source": "test_source"}]]
        }
        mock_genai.GenerativeModel.return_value.generate_content.return_value = MagicMock(text="モック回答")
        request_body = {"user_id": "test_user", "message": "テスト質問"}
        response = client.post("/chat", json=request_body)
        assert response.status_code == 200
        assert response.json()["response"] == "モック回答"
        mock_qa_collection.query.assert_called_once()
        mock_genai.GenerativeModel.return_value.generate_content.assert_called_once()

@patch('backend.main.genai')
@patch('backend.main.qa_collection')
def test_chat_endpoint_no_context(mock_qa_collection, mock_genai):
    """
    ChromaDBに関連情報が見つからなかった場合のテスト。
    """
    with patch('backend.main.get_embedding', return_value=[0.1, 0.2, 0.3]):
        mock_qa_collection.query.return_value = {"documents": [[]]}
        mock_genai.GenerativeModel.return_value.generate_content.return_value = MagicMock(text="関連情報はありません。")
        request_body = {"user_id": "test_user", "message": "関連のない質問"}
        response = client.post("/chat", json=request_body)
        assert response.status_code == 200
        assert response.json()["response"] == "関連情報はありません。"
        
@patch('backend.main.genai')
def test_chat_endpoint_gemini_api_error(mock_genai):
    """
    Gemini API呼び出しでエラーが発生した場合のテスト。
    """
    with patch('backend.main.get_embedding', return_value=[0.1, 0.2, 0.3]):
        mock_genai.GenerativeModel.return_value.generate_content.side_effect = Exception("API error")
        request_body = {"user_id": "test_user", "message": "テスト質問"}
        response = client.post("/chat", json=request_body)
        assert response.status_code == 500
        assert "An internal server error occurred" in response.json()["detail"]