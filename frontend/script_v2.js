document.addEventListener('DOMContentLoaded', () => {
  const chatMessages = document.getElementById('chat-messages');
  const userInput = document.getElementById('user-input');
  const sendButton = document.getElementById('send-button');

  let scenarioPath = ["root"];

  // =========================================================
  // API_BASE を確定（/chat は付けない）
  // - index.html で window.__API_BASE を注入していればそれを優先
  // - なければ hostname で localhost / deploy を切り替え
  // =========================================================
  const currentHostname = window.location.hostname;

  const API_BASE =
    (typeof window.__API_BASE === 'string' && window.__API_BASE.trim() !== '')
      ? window.__API_BASE.trim()
      : ((currentHostname === 'localhost' || currentHostname === '127.0.0.1')
          ? 'http://localhost:8000'
          : 'https://chatbot-backend-140594287961.asia-northeast1.run.app');

  // デバッグ用：ブラウザ側で確認できるように露出（注入用とは別名）
  window.__DEBUG_API_BASE = API_BASE;
  console.log("API_BASE:", API_BASE);

  // 初期化
  initScenario();

  // イベント
  sendButton.addEventListener('click', sendMessage);
  userInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') sendMessage();
  });

  // =========================================================
  // UI helper
  // =========================================================
  function appendMessage(sender, text) {
    const messageElement = document.createElement('div');
    messageElement.classList.add('message', sender);

    const contentElement = document.createElement('div');
    contentElement.classList.add('message-content');
    contentElement.innerHTML = String(text ?? '').replace(/\n/g, '<br>');

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

  function appendOptions(options) {
    if (!options || options.length === 0) return;

    // 1) category/select と nav を分ける
    const navOpts = options.filter(o => o.action === 'nav');
    const mainOpts = options.filter(o => o.action !== 'nav'); // selectなど

    // 2) ラップ（2段構成）
    const container = document.createElement('div');
    container.classList.add('options-container');

    const mainRow = document.createElement('div');
    mainRow.classList.add('options-wrap'); // 既存の横並びスタイルを流用

    const navRow = document.createElement('div');
    navRow.classList.add('nav-wrap'); // 新規（小さめ）

    // ボタン生成関数
    const makeBtn = (opt, isNav) => {
      const btn = document.createElement('button');
      btn.classList.add(isNav ? 'nav-button' : 'option-button');
      btn.textContent = opt.label;

      btn.addEventListener('click', () => {
        appendMessage('user', opt.label);
        sendScenarioSelect(opt.id); // navでもselectでも同じAPI仕様ならこれでOK
      });

      return btn;
    };

    mainOpts.forEach(opt => mainRow.appendChild(makeBtn(opt, false)));
    navOpts.forEach(opt => navRow.appendChild(makeBtn(opt, true)));

    container.appendChild(mainRow);
    if (navOpts.length) container.appendChild(navRow);

    // ボット側メッセージとして表示
    const messageElement = document.createElement('div');
    messageElement.classList.add('message', 'bot');

    const contentElement = document.createElement('div');
    contentElement.classList.add('message-content');
    contentElement.appendChild(container);

    messageElement.appendChild(contentElement);
    chatMessages.appendChild(messageElement);
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }

  // =========================================================
  // API calls
  // =========================================================
  async function initScenario() {
    try {
      const res = await fetch(`${API_BASE}/init`, {
        method: "GET",
        cache: "no-store",
      });

      if (!res.ok) {
        const t = await res.text();
        throw new Error(`HTTP ${res.status}: ${t}`);
      }

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
    // 入力を無効化（連打防止）
    sendButton.disabled = true;
    userInput.disabled = true;

    const loadingMessageId = 'loading-msg-' + Date.now();
    appendLoadingMessage(loadingMessageId);

    const userId = 'anonymous_user';

    try {
      const res = await fetch(`${API_BASE}/scenario/select`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          node_id: nodeId,
          path: scenarioPath,
          user_id: userId
        }),
      });

      if (!res.ok) {
        const t = await res.text();
        throw new Error(`HTTP ${res.status}: ${t}`);
      }

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
      const res = await fetch(`${API_BASE}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message, user_id: userId }),
      });

      if (!res.ok) {
        const t = await res.text();
        throw new Error(`HTTP ${res.status}: ${t}`);
      }

      const data = await res.json();
      removeLoadingMessage(loadingMessageId);
      appendMessage('bot', data.response);

      // もし自由入力で path を返す仕様なら反映（任意）
      if (data.ui && data.ui.path) scenarioPath = data.ui.path;

      // もし自由入力でも options を返すなら表示（任意）
      if (data.options) appendOptions(data.options);

    } catch (e) {
      console.error('chat failed:', e);
      removeLoadingMessage(loadingMessageId);
      appendMessage('bot', 'エラーが発生しました。もう一度お試しください。');
    } finally {
      sendButton.disabled = false;
      userInput.disabled = false;
      userInput.focus();
    }
  }
});
