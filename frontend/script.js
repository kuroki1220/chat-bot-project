document.addEventListener('DOMContentLoaded', () => {
    const chatMessages = document.getElementById('chat-messages');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');

    // 中間サーバーのエンドポイントURLを動的に設定
    let API_ENDPOINT;
    const currentHostname = window.location.hostname;

    if (currentHostname == 'localhost' || currentHostname === '127.0.0.1'){
        //ローカル環境の場合
        API_ENDPOINT = 'http://localhost:8000/chat';
        console.log("ローカル環境を検出しました。APIエンドポイント: " + API_ENDPOINT);
    }else{
        //Cloud Runなどのデプロイ環境の場合
        //自分のCloud RunのService URLを記載
        API_ENDPOINT = 'https://chatbot-backend-445070682859.asia-northeast1.run.app/chat';
        console.log("デプロイ環境を検出しました。APIエンドポイント: " + API_ENDPOINT);
    }

    // 初期メッセージを表示
    appendMessage('bot', 'こんにちは！どのようなご質問がありますか？');

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
            const response = await fetch(API_ENDPOINT, {
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