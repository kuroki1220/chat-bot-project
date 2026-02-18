# chat-bot-system/backend/Dockerfile

# Pythonの公式イメージをベースにする
#FROM python:3.11-slim-buster  #古いリポジトリのためコメントアウト
#新しい安定板のDebian(bookworm)を使用
FROM python:3.11-slim-bookworm  

# apt-getを更新し、必要なビルドツールをインストール
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libsqlite3-dev \
    libffi-dev \
    zlib1g-dev \
    sqlite3 \
    wget \
    file \
    && rm -rf /var/lib/apt/lists/*

#最新のSQLite3をダウンロード、ビルド、インストール
ENV SQLITE_VERSION 3.46.0
RUN wget -q https://www.sqlite.org/2024/sqlite-autoconf-3460000.tar.gz && \
    tar xzf sqlite-autoconf-3460000.tar.gz && \
    cd sqlite-autoconf-3460000 && \
    ./configure --prefix=/usr/local && \
    make && \
    make install && \
    ldconfig && \
    cd .. && \
    rm -rf sqlite-autoconf-3460000.tar.gz sqlite-autoconf-3460000

#新しいSQLite3ライブラリへのパスを設定
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

# 作業ディレクトリを設定
WORKDIR /app

# 依存関係ファイルをコピーしてインストール
COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションコードをコピー
COPY . .

# 環境変数 (Gemini APIキー) を設定
# 本番環境ではKMSやSecret Managerなどで安全に管理することを推奨
# Cloud Runでは、デプロイ時に環境変数を直接設定できます
# ENV GEMINI_API_KEY="YOUR_GEMINI_API_KEY" # Dockerfileには直接書かないのがベスト

# アプリケーションを実行するコマンドを定義
# Uvicornを使って、ポート8000でアプリケーションを起動
# Cloud Runは環境変数 PORT でポートを指定するため、それを使用
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]