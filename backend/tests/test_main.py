# backend/tests/test_main.py

import pytest
import sqlite3
import os
import tempfile
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from pathlib import Path

# テスト対象のmain.pyから関数やappインスタンスをインポート
# 注: import 文は、プロジェクトの構成に合わせて変更されています。
from backend.main import (
    app,
    save_query_log,
    LOG_DB_PATH,
    init_log_db,
    ChatRequest,
    get_embedding
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
def test_save_query_log_success():
    """
    save_query_log関数がログを正しく保存するかテストする。
    """
    # 一時的なディレクトリを作成し、そのパスをモックする
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_db_path = Path(temp_dir) / "test_chatbot_logs.db"

        #main.py内のinit_log_db関数とLOG_DB_PATH変数をモック
        with patch('backend.main.LOG_DB_PATH', str(temp_db_path)):
            # データベースを初期化
            init_log_db()

            user_id = "test_user_001"
            query = "テスト質問"
            response = "テスト回答"
            context = "テストコンテキスト"
            save_query_log(user_id, query, response, context)

            # モックされたパスを使ってコネクションを確立し、検証する
            conn = sqlite3.connect(temp_db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT user_id, query_text, bot_response_text, context_used FROM queries WHERE user_id = ?", (user_id,))
            result = cursor.fetchone()
            conn.close()

            # 結果が期待通りかアサート
            assert result is not None
            assert result[0] == user_id
            assert result[1] == query
            assert result[2] == response
            assert result[3] == context

# --- 内部結合テスト ---
@patch('backend.main.genai')
@patch('backend.main.qa_collection')
def test_chat_endpoint_success(mock_qa_collection, mock_genai):
    """
    /chatエンドポイントがRAGプロセス全体を通して正しく動作するかをテストする。
    """
    # モックのセットアップ
    # 1. get_embedding関数のモック
    with patch('backend.main.get_embedding', return_value=[0.1, 0.2, 0.3]):
        # 2. ChromaDBのクエリ結果のモック
        mock_qa_collection.query.return_value = {
            "documents": [["これはQ&Aデータから取得したコンテキストです。"]],
            "metadatas": [[{"source": "test_source"}]]
        }

        # 3. Gemini APIの応答のモック
        mock_genai.GenerativeModel.return_value.generate_content.return_value = MagicMock(text="モック回答")

        # テスト用のリクエストボディ
        request_body = {"user_id": "test_user", "message": "テスト質問"}
        response = client.post("/chat", json=request_body)

        # アサーション
        assert response.status_code == 200
        assert response.json()["response"] == "モック回答"

        # 関数が正しく呼び出されたか検証
        mock_qa_collection.query.assert_called_once()
        mock_genai.GenerativeModel.return_value.generate_content.assert_called_once()

@patch('backend.main.genai')
@patch('backend.main.qa_collection')
def test_chat_endpoint_no_context(mock_qa_collection, mock_genai):
    """
    ChromaDBに関連情報が見つからなかった場合のテスト。
    """
    with patch('backend.main.get_embedding', return_value=[0.1, 0.2, 0.3]):
        # ChromaDBが空の結果を返すようモック
        mock_qa_collection.query.return_value = {"documents": [[]]}

        # Gemini APIの応答をモック
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
        # Gemini API呼び出し時に例外を発生させるようモック
        mock_genai.GenerativeModel.return_value.generate_content.side_effect = Exception("API error")

        request_body = {"user_id": "test_user", "message": "テスト質問"}
        response = client.post("/chat", json=request_body)

        # 500エラーが返されることを確認
        assert response.status_code == 500
        assert "An internal server error occurred" in response.json()["detail"]
