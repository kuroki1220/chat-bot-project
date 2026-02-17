document.addEventListener('DOMContentLoaded', () => {
    const chatMessages = document.getElementById('chat-messages');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');

    let scenarioPath = ["root"]; //バックエンドに渡すパス

    // 中間サーバーのエンドポイントURLを動的に設定
    // APIのベースURL（末尾に /chat を付けない）
    let API_BASE;
    const currentHostname = window.location.hostname;

    if (currentHostname === 'localhost' || currentHostname === '127.0.0.1') {
        // ローカル環境
        API_BASE = 'http://localhost:8000';
        console.log("ローカル環境を検出しました。API_BASE: " + API_BASE);
    } else {
        // デプロイ環境（Cloud Run）
        API_BASE = 'https://chatbot-backend-140594287961.asia-northeast1.run.app';
        console.log("デプロイ環境を検出しました。API_BASE: " + API_BASE);
    }

    initScenario();

    sendButton.addEventListener('click', sendMessage);
    userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });

    function appendMessage(sender, text) {
        const messageElement = document.createElement('div');
        messageElement.classList.add('message', sender);

        const contentElement = document.createElement('div');
        contentElement.classList.add('message-content');
        contentElement.innerHTML = text.replace(/\n/g, '<br>'); // 改行コードを <br> タグに変換

        messageElement.appendChild(contentElement);
        chatMessages.appendChild(messageElement);

        // 最新のメッセージが見えるようにスクロール
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    async function initScenario() {
        try {
            const res = await fetch(`${API_BASE}/init`);
            const data = await res.json();

            appendMessage('bot', data.response);
            if (data.ui && data.ui.path) scenarioPath = data.ui.path;

            appendOptions(data.options);
        } catch (e) {
            console.error("init failed", e);
            appendMessage('bot', '初期化に失敗しました。ページを再読み込みしてください。');
        }
    }

    async function sendScenarioSelect(nodeId) {
        // 入力を無効化 (連打防止)
        sendButton.disabled = true;
        userInput.disabled = true;

        const loadingMessageId = 'loading-msg-' + Date.now();
        appendLoadingMessage(loadingMessageId);

        const userId = 'anonymous_user';

        try {
            const res = await fetch(`${API_BASE}/scenario/select`, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    node_id: nodeId,
                    path: scenarioPath,
                    user_id: userId
                })
            });

            const data = await res.json();
            removeLoadingMessage(loadingMessageId);

            appendMessage('bot', data.response);

            if (data.ui && data.ui.path) scenarioPath = data.ui.path;

            appendOptions(data.options);
        } catch (e) {
            console.error("scenario select failed", e);
            removeLoadingMessage(loadingMessageId);
            appendMessage('bot', 'エラーが発生しました。もう一度お試しください。');
        } finally {
            sendButton.disabled = false;
            userInput.disabled = false;
            userInput.focus();
        }
    }

    async function sendMessage() {
        const message = userInput.value.trim();
        if (message === '') {
            return; // 空のメッセージは送信しない
        }

        appendMessage('user', message); // ユーザーのメッセージを表示
        userInput.value = ''; // 入力フィールドをクリア
        sendButton.disabled = true; // 送信ボタンを無効化
        userInput.disabled = true; // 入力フィールドを無効化

        // ローディングメッセージを表示
        const loadingMessageId = 'loading-msg-' + Date.now();
        appendLoadingMessage(loadingMessageId);
    
        // 仮のユーザーIDを設定
        const userId = 'anonymous_user';

        try {
            const response = await fetch(`${API_BASE}/chat`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                // user_idをリクエストボディに追加
                body: JSON.stringify({ message: message, user_id: userId }),
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(`サーバーエラー: ${response.status} - ${errorData.detail || response.statusText}`);
            }

            const data = await response.json();
            removeLoadingMessage(loadingMessageId); // ローディングメッセージを削除
            appendMessage('bot', data.response); // ボットの返答を表示

        } catch (error) {
            console.error('チャットの送信中にエラーが発生しました:', error);
            removeLoadingMessage(loadingMessageId); // ローディングメッセージを削除
            appendMessage('bot', 'エラーが発生しました。もう一度お試しください。');
        } finally {
            sendButton.disabled = false; // 送信ボタンを有効化
            userInput.disabled = false; // 入力フィールドを有効化
            userInput.focus(); // 入力フィールドにフォーカスを戻す
        }
    }

    function appendOptions(options) {
        if (!options || options.length === 0) return;

        const wrap = document.createElement('div');
        wrap.classList.add('options-wrap');

        options.forEach(opt => {
            const btn = document.createElement('button');
            btn.classList.add('option-button');
            btn.textContent = opt.label;

            btn.addEventListener('click', () => {
                // クリックした文言をユーザー吹き出しに出す
                appendMessage('user', opt.label);

                // nav系の押下の場合は path を送る
                if (opt.action === 'nav') {
                    sendScenarioSelect(opt.id);
                    return;
                }

                // category/answer 選択
                sendScenarioSelect(opt.id);
            });

            wrap.appendChild(btn);
        });

        // ボット側メッセージとして表示する（左寄せに合わせたいので message.bot を付ける）
        const messageElement = document.createElement('div');
        messageElement.classList.add('message', 'bot');

        const contentElement = document.createElement('div');
        contentElement.classList.add('message-content');
        contentElement.appendChild(wrap);

        messageElement.appendChild(contentElement);
        chatMessages.appendChild(messageElement);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    // ローディングメッセージの表示
    function appendLoadingMessage(id) {
        const messageElement = document.createElement('div');
        messageElement.classList.add('message', 'bot', 'loading');
        messageElement.id = id;

        const contentElement = document.createElement('div');
        contentElement.classList.add('message-content');
        contentElement.textContent = '思考中...'; // またはローディングスピナーなど

        messageElement.appendChild(contentElement);
        chatMessages.appendChild(messageElement);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    // ローディングメッセージの削除
    function removeLoadingMessage(id) {
        const loadingMessage = document.getElementById(id);
        if (loadingMessage) {
            loadingMessage.remove();
        }
    }
});