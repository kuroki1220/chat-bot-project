document.addEventListener('DOMContentLoaded', () => {
  const chatMessages = document.getElementById('chat-messages');
  const userInput = document.getElementById('user-input');
  const sendButton = document.getElementById('send-button');

  let scenarioPath = ["root"];

  // API_BASE を確定させる（/chat は付けない）
  const currentHostname = window.location.hostname;

  const API_BASE =
    (currentHostname === 'localhost' || currentHostname === '127.0.0.1')
      ? 'http://localhost:8000'
      : 'https://chatbot-backend-140594287961.asia-northeast1.run.app';

  // デバッグ用：ブラウザ側で確認できるように露出
  window.__API_BASE = API_BASE;

  console.log("API_BASE:", API_BASE);

  initScenario();

  sendButton.addEventListener('click', sendMessage);
  userInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') sendMessage();
  });

  function appendMessage(sender, text) {
    const messageElement = document.createElement('div');
    messageElement.classList.add('message', sender);

    const contentElement = document.createElement('div');
    contentElement.classList.add('message-content');
    contentElement.innerHTML = String(text).replace(/\n/g, '<br>');

    messageElement.appendChild(contentElement);
    chatMessages.appendChild(messageElement);
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }

  async function initScenario() {
    try {
      const res = await fetch(`${API_BASE}/init`, { cache: "no-store" });
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
    sendButton.disabled = true;
    userInput.disabled = true;

    const loadingMessageId = 'loading-msg-' + Date.now();
    appendLoadingMessage(loadingMessageId);

    const userId = 'anonymous_user';

    try {
      const res = await fetch(`${API_BASE}/scenario/select`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ node_id: nodeId, path: scenarioPath, user_id: userId }),
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
    if (!message) return;

    appendMessage('user', message);
    userInput.value = '';
    sendButton.disabled = true;
    userInput.disabled = true;

    const loadingMessageId = 'loading-msg-' + Date.now();
    appendLoadingMessage(loadingMessageId);

    const userId = 'anonymous_user';

    try {
      const response = await fetch(`${API_BASE}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message, user_id: userId }),
      });

      if (!response.ok) {
        // ここで 500/403/404 の内容が見えるようにする
        const t = await response.text();
        throw new Error(`HTTP ${response.status}: ${t}`);
      }

      const data = await response.json();
      removeLoadingMessage(loadingMessageId);
      appendMessage('bot', data.response);
    } catch (error) {
      console.error('chat failed:', error);
      removeLoadingMessage(loadingMessageId);
      appendMessage('bot', 'エラーが発生しました。もう一度お試しください。');
    } finally {
      sendButton.disabled = false;
      userInput.disabled = false;
      userInput.focus();
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
        appendMessage('user', opt.label);
        sendScenarioSelect(opt.id);
      });

      wrap.appendChild(btn);
    });

    const messageElement = document.createElement('div');
    messageElement.classList.add('message', 'bot');

    const contentElement = document.createElement('div');
    contentElement.classList.add('message-content');
    contentElement.appendChild(wrap);

    messageElement.appendChild(contentElement);
    chatMessages.appendChild(messageElement);
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }

  function appendLoadingMessage(id) {
    const messageElement = document.createElement('div');
    messageElement.classList.add('message', 'bot', 'loading');
    messageElement.id = id;

    const contentElement = document.createElement('div');
    contentElement.classList.add('message-content');
    contentElement.textContent = '思考中...';

    messageElement.appendChild(contentElement);
    chatMessages.appendChild(messageElement);
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }

  function removeLoadingMessage(id) {
    const loadingMessage = document.getElementById(id);
    if (loadingMessage) loadingMessage.remove();
  }
});
