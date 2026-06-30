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
          : 'https://chatbot-backend-685448718484.asia-northeast1.run.app');

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

  const LINK_PREVIEWS = {

  "": {
    title: "｜",
    description: "",
    domain: "",
    image: ""
  },

  "https://bb-fletsnavi.com/westlp/nttwest/index.html": {
    title: "フレッツ光 サービス概要｜NTT西日本｜フレッツ光 / NTT豪華キャンペーンお申込みサイト",
    description: "フレッツ光は最大10Gbps高速インターネット！NTTフレッツ光豪華特典で大幅割引！",
    domain: "bb-fletsnavi.com",
    image: "https://th.bing.com/th/id/R.f8cd1860fec07065d621938f06b2d7ea?rik=YOo8%2bxPIgFLFOA&riu=http%3a%2f%2fbb-navi.jp%2fimage%2fwest%2fetc%2fflets_logo.gif&ehk=qzA3xSEydew0HH0vHV1nwiM%2fjNFEiJL1zZs1N40aVAw%3d&risl=&pid=ImgRaw&r=0"
  },

  "https://flets-w.com/collabo/tenyou/entry/": {
    title: "転用承諾番号のお受け取り｜光コラボレーションモデル｜フレッツ光｜NTT西日本公式",
    description: "転用お手続きに関するご案内ページ。転用承諾番号のお受け取りについてご案内します。",
    domain: "flets-w.com",
    image: "https://www.ntt-west.co.jp/news/image/ogp/ogp_westci.png"
  },

  "https://flets.com/app4/input/index/": {
    title: "NTT東日本｜光アクセスページへの移行（転用）のお手続き｜フレッツ光",
    description: "光コラボレーション事業者さまが提供する、光アクセスサービスへの移行（転用）のお手続きページです。",
    domain: "flets.com",
    image: "https://www.ntt-east.co.jp/release/detail/images/img_20250509_02_02.png"
  },

  "https://flets.com/collabo/list/?utm_source=chatgpt.com": {
    title: "「光コラボレーション事業者さま」及び「お取り扱いサービス」一覧｜光コラボレーションモデル",
    description: "【光コラボレーションモデル】について、「光コラボレーション事業者さま」及び「お取り扱いサービス」の一覧を掲載しています。",
    domain: "flets.com",
    image: "https://th.bing.com/th/id/OIP.dbrxn6yzv6APdNU86XXbVQHaEK?o=7rm=3&rs=1&pid=ImgDetMain&o=7&rm=3"
  },

  "https://flets-w.com/cart/index.php": {
    title: "NTT西日本公式｜フレッツ光｜光インターネット接続（光回線）",
    description: "",
    domain: "flets.ntt-west.co.jp",
    image: "https://www.ntt-west.co.jp/news/image/ogp/ogp_westci.png"
  },

  "https://flets.com/app2/search_c.html": {
    title: "光アクセスサービス 提供エリア検索｜フレッツ光公式｜NTT東日本",
    description: "",
    domain: "flets.com",
    image: "https://www.ntt-east.co.jp/release/detail/images/img_20250509_02_02.png"
  },

  "https://www.soumu.go.jp/menu_seisaku/ictseisaku/telephonerelay/index.html": {
    title: "総務省｜聴覚障碍者等の電話利用の円滑化",
    description: "",
    domain: "www.soumu.go.jp",
    image: "https://logomarket.jp/labo/wp-content/uploads/2015/07/b1e55ec9e6bc3930762432b10434d0c2.jpg"
  },

  "https://www.soumu.go.jp/main_sosiki/joho_tsusin/universalservice/": {
    title: "総務省｜ユニバーサルサービス制度",
    description: "",
    domain: "www.soumu.go.jp",
    image: "https://logomarket.jp/labo/wp-content/uploads/2015/07/b1e55ec9e6bc3930762432b10434d0c2.jpg"
  },

  "https://www.ntt-west.co.jp/denwa/voicewp/": {
    title: "ボイスワープの設定方法｜固定電話・加入電話｜NTT西日本",
    description: "ボイスワープの設定方法に関するご案内です。加入電話、INSネット、ひかり電話のそれぞれの設定方法についてご確認いただけます。",
    domain: "www.ntt-west.co.jp",
    image: "https://tse3.mm.bing.net/th/id/OIP.olpG_LQiWExKfAV1n-0DOwHaDT?rs=1&pid=ImgDetMain&o=7&rm=3"
  },

  "https://flets-w.com/opt/ftv/terms/channel.html": {
    title: "チャンネル一覧｜フレッツ・テレビ｜NTT西日本公式｜フレッツ光｜光インターネット接続",
    description: "フレッツ・テレビのチャンネル一覧をご案内します。",
    domain: "flets-w.com",
    image: "https://www.ntt-west.co.jp/news/image/ogp/ogp_westci.png"
  },

  "https://flets.com/ftv/channel.html": {
    title: "【公式】NTT東日本｜フレッツ光｜フレッツ・テレビ｜視聴可能チャンネル｜個人のお客様",
    description: "フレッツ・テレビの視聴可能チャンネル。",
    domain: "flets.com",
    image: "https://www.ntt-east.co.jp/release/detail/images/img_20250509_02_02.png"
  },

  "https://www.slim-j.net/mypage/": {
    title: "ログイン",
    description: "",
    domain: "www.slim-j.net",
    image: "https://internshipguide.jp/img/corporate_logo/0000_4916/jpr1Snc.png"
  },

  "https://www.ntt-west.co.jp/kiki/support/hgw_lamp/": {
    title: "【NTT西日本】ひかり電話対応機器・ホームゲートウェイのランプ状態 - 情報・通信機器",
    description: "ひかり電話対応機器・ホームゲートウェイのランプ状態について。",
    domain: "www.ntt-west.co.jp",
    image: "https://www.ntt-west.co.jp/news/image/ogp/ogp_westci.png"
  },

  "https://flets-w.com/user/support/solve/notconnect/check_03/": {
    title: "【NTT西日本】設定・トラブルサポート Web 113",
    description: "NTT西日本公式の故障受付サイト(Web113)です。",
    domain: "www.ntt-west.co.jp",
    image: "https://www.ntt-west.co.jp/news/image/ogp/ogp_westci.png"
  },

  "https://web113.ntt-east.co.jp/selfcheck/internet/": {
    title: "故障に関するよくある質問 Web 113｜NTT東日本｜フレッツ光",
    description: "困ったことがあれば、こちらのページをご覧ください。",
    domain: "faq.web113.ntt-east.co.jp",
    image: "https://www.ntt-east.co.jp/release/detail/images/img_20250509_02_02.png"
  },

  "https://www.ntt-west.co.jp/kiki/download/flets/pr500ki/PR-500KI_detail/guide/6-i/m06_m57.html": {
    title: "電話で設定するー電話機で設定する",
    description: "電話機からひかり電話の設定をするには",
    domain: "www.ntt-west.co.jp",
    image: "https://th.bing.com/th/id/OIP.dbrxn6yzv6APdNU86XXbVQHaEK?o=7rm=3&rs=1&pid=ImgDetMain&o=7&rm=3"
  },

  "https://faq.web113.ntt-east.co.jp/%E3%81%94%E8%87%AA%E5%AE%85%E3%81%AE%E9%9B%BB%E8%A9%B1%E6%A9%9F%E3%82%92%E4%BD%BF%E3%81%A3%E3%81%A6%E3%81%B2%E3%81%8B%E3%82%8A%E9%9B%BB%E8%A9%B1%E3%81%AE%E9%B3%B4%E3%82%8A%E5%88%86%E3%81%91%E3%82%92%E8%A8%AD%E5%AE%9A%E3%82%92%E3%81%99%E3%82%8B-6889aea9826ffbf60579260b": {
    title: "ご自宅の電話機を使ってひかり電話のなり分けの設定をする - 故障に関するよくある質問 Web 113｜NTT東日本｜フレッツ光",
    description: "ご自宅の電話機を使ってひかり電話のなり分けの設定をする",
    domain: "faq.web113.ntt-east.co.jp",
    image: "https://th.bing.com/th/id/OIP.dbrxn6yzv6APdNU86XXbVQHaEK?o=7rm=3&rs=1&pid=ImgDetMain&o=7&rm=3"
  },

  "https://support.microsoft.com/ja-jp/office/outlook-for-windows-%E3%81%AB%E3%83%A1%E3%83%BC%E3%83%AB-%E3%82%A2%E3%82%AB%E3%82%A6%E3%83%B3%E3%83%88%E3%82%92%E8%BF%BD%E5%8A%A0%E3%81%99%E3%82%8B-6e27792a-9267-4aa4-8bb6-c84ef146101b": {
    title: "Outlook for Windows にメールアカウントを追加する - Microsoft サポート",
    description: "Microsoft 365、POP、IMAP、または Microsoft Exchange ベースのメールアカウントを利用できるように Outlook を設定する方法。",
    domain: "support.microsoft.com",
    image: "https://th.bing.com/th/id/OIP.YGpwvn0lkOZIt90a7NLEYQHaHa?o=7rm=3&rs=1&pid=ImgDetMain&o=7&rm=3"
  },

  "https://support.google.com/mail/answer/56256?hl=ja": {
    title: "Gmail アカウントの作成 - Gmail ヘルプ",
    description: "Gmail をお使いになるには、Google アカウントを作成してください。",
    domain: "support.google.com",
    image: "https://th.bing.com/th/id/R.0fa3fe04edf6c0202970f2088edea9e7?rik=joOK76LOMJlBPw&riu=http%3a%2f%2fpluspng.com%2fimg-png%2fgoogle-logo-png-open-2000.png&ehk=0PJJlqaIxYmJ9eOIp9mYVPA4KwkGo5Zob552JPltDMw%3d&risl=&pid=ImgRaw&r=0"
  },

  "https://www.wikihow.jp/mozilla-thunderbird%E3%82%92%E3%82%BB%E3%83%83%E3%83%88%E3%82%A2%E3%83%83%E3%83%97%E3%81%99%E3%82%8B": {
    title: "Thunderbirdメールアカウントの設定方法：完全ガイド",
    description: "",
    domain: "www.wikihow.jp",
    image: "https://logos-world.net/wp-content/uploads/2023/06/Mozilla-Thunderbird-Logo-500x281.png"
  },

  "https://support.apple.com/ja-jp/guide/mail/mlhl5094a9f2/16.0/mac/26": {
    title: "Macで「メール」を使い始める - Apple サポート（日本）",
    description: "Macの「メール」をすぐに使い始めるための基本操作について説明します。",
    domain: "support.apple.com",
    image: "https://logospng.org/download/apple/logo-apple-1536.png"
  },

  "https://support.apple.com/ja-jp/102619": {
    title: "iPhoneやiPadにメールアカウントを追加する - Apple サポート（日本）",
    description: "iOSデバイスのメールアプリでメールアカウントを自動または手動で設定します。",
    domain: "support.apple.com",
    image: "https://logospng.org/download/apple/logo-apple-1536.png"
  },

  "https://support.google.com/mail/answer/6078445?hl=ja": {
    title: "Gmail アプリに別のメールアカウントを追加する - Android - Gmail ヘルプ",
    description: "Android 版、iPhone 版、iPad 版の Gmail アプリでは、次のものを追加できます。",
    domain: "support.google.com",
    image: "https://th.bing.com/th/id/R.0fa3fe04edf6c0202970f2088edea9e7?rik=joOK76LOMJlBPw&riu=http%3a%2f%2fpluspng.com%2fimg-png%2fgoogle-logo-png-open-2000.png&ehk=0PJJlqaIxYmJ9eOIp9mYVPA4KwkGo5Zob552JPltDMw%3d&risl=&pid=ImgRaw&r=0"
  },

  "https://support.microsoft.com/ja-jp/windows/windows-%E3%81%A7-wi-fi-%E3%83%8D%E3%83%83%E3%83%88%E3%83%AF%E3%83%BC%E3%82%AF%E3%81%AB%E6%8E%A5%E7%B6%9A%E3%81%99%E3%82%8B-1f881677-b569-0cd5-010d-e3cd3579d263": {
    title: "Windows で Wi-Fi ネットワークに接続する - Microsoft サポート",
    description: "Windows で Wi-Fi ネットワークに接続し、現在のネットワーク接続を管理する方法について説明します。",
    domain: "support.microsoft.com",
    image: "https://th.bing.com/th/id/OIP.YGpwvn0lkOZIt90a7NLEYQHaHa?o=7rm=3&rs=1&pid=ImgDetMain&o=7&rm=3"
  },

  "https://support.google.com/android/answer/9075847?hl=ja": {
    title: "Android デバイスを Wi-fi ネットワークに接続する - Android ヘルプ",
    description: "自分に合った方法で Wi-Fi を使用するために、デバイスを接続する方法とタイミングを変更できます。",
    domain: "support.google.com",
    image: "https://th.bing.com/th/id/R.0fa3fe04edf6c0202970f2088edea9e7?rik=joOK76LOMJlBPw&riu=http%3a%2f%2fpluspng.com%2fimg-png%2fgoogle-logo-png-open-2000.png&ehk=0PJJlqaIxYmJ9eOIp9mYVPA4KwkGo5Zob552JPltDMw%3d&risl=&pid=ImgRaw&r=0"
  },

  "https://support.apple.com/ja-jp/111107": {
    title: "iPhoneまたはiPadでWi-Fiに接続する - Apple サポート（日本）",
    description: "公開ネットワーク、保護されたネットワーク、過去に接続したことがあるネットワークなど、デバイスをWi-Fiネットワークに接続する。",
    domain: "support.apple.com",
    image: "https://logospng.org/download/apple/logo-apple-1536.png"
  },

    "https://bb-fletsnavi.com/westlp/nttwest/index.html?vd=g1_5_2&gad_source=1&gad_campaignid=22613274467&gbraid=0AAAAA_yDh7HgPrwXQE2cYcWahoCyurNme&gclid=CjwKCAjwiY_GBhBEEiwAFaghvtjmVM4oU-xi7ssgbMYWBRYK9are89Qbfbe5Lxl9ClEqKatp88o05hoCSl0QAvD_BwE":{
    title: "フレッツ光 サービス概要 | NTT西日本 | フレッツ光／NTT豪華キャンペーンお申し込みサイト",
    description: "フレッツ光は最大10Gbps高速インターネット！NTTフレッツ光豪華特典で大幅割引！さらに当サイト特典で最大79,000円現金キャッシュバック。",
    domain: "bb-fletsnavi.com",
    image: "https://logospng.org/download/apple/logo-apple-1536.png"
  }

};

  // =========================================================
  // UI helper
  // =========================================================
  function formatMessageText(text) {
    let formatted = String(text ?? '');

    // [表示文字](URL) をリンク化
    formatted = formatted.replace(
      /\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g,
      (match, label, url) => {
        const preview = LINK_PREVIEWS[url];

        if (!preview) {
          return `<a href="${url}" target="_blank" rel="noopener noreferrer" class="chat-link">${label}</a>`;
        }

        const imageHtml = preview.image
          ? `<div class="link-preview-image"><img src="${preview.image}" alt=""></div>`
          : "";

        return `
          <a href="${url}" target="_blank" rel="noopener noreferrer" class="link-preview-card">${imageHtml}<div class="link-preview-body"><div class="link-preview-title">${preview.title}</div><div class="link-preview-description">${preview.description}</div><div class="link-preview-domain">${preview.domain}</div></div></a>`;
      }
    );

    formatted = formatted.replace(
      /(?<!["(])(https?:\/\/[^\s<]+)/g,
      (url) => {
        const cleanUrl = url.replace(/[。、)]$/, "");
        const preview = LINK_PREVIEWS[cleanUrl];

        if (!preview) {
          return `<a href="${cleanUrl}" target="_blank" rel="noopener noreferrer" class="chat-link">${cleanUrl}</a>`;
        }

        const imageHtml = preview.image
          ? `<div class="link-preview-image"><img src="${preview.image}" alt=""></div>`
          : "";

        return `<a href="${cleanUrl}" target="_blank" rel="noopener noreferrer" class="link-preview-card">${imageHtml}<div class="link-preview-body"><div class="link-preview-title">${preview.title}</div><div class="link-preview-description">${preview.description}</div><div class="link-preview-domain">${preview.domain}</div></div></a>`;
      }
    );

    // CSVから来る「文字としての \n」を改行にする
    formatted = formatted.replace(/\\n/g, '<br>');

    // 本物の改行コードも <br> にする
    formatted = formatted.replace(/\n/g, '<br>');

    return formatted;
  }
  
  function appendMessage(sender, text, extraClass = '') {
    const messageElement = document.createElement('div');
    messageElement.classList.add('message', sender);

    if (extraClass) {
      messageElement.classList.add(extraClass);
    }

    const contentElement = document.createElement('div');
    contentElement.classList.add('message-content');
    contentElement.innerHTML = formatMessageText(text);

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
    contentElement.innerHTML = `
      <div class="loading-wave">
        <span>思</span><span>考</span><span>中</span>
        <div class="loading-dots">
          <i></i><i></i><i></i>
        </div>
      </div>
    `;

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
        if (!isNav) {
          btn.classList.add('selected');
        }

        appendMessage('user', opt.label);
        sendScenarioSelect(opt.id);
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

    showInitialLoading();

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

      hideInitialLoading();

      appendMessage('bot', data.response, 'wide-message');
      if (data.ui && data.ui.path) scenarioPath = data.ui.path;

      appendOptions(data.options);
    } catch (e) {
      console.error("init failed", e);

      hideInitialLoading();

      appendMessage('bot', '初期化に失敗しました。ページを再読み込みしてください。');
    }
  }
  function showInitialLoading() {
      chatMessages.innerHTML = `
        <div class="initial-loader">
          <div class="simple-line-loader">
            <span></span>
            <span></span>
            <span></span>
            <p>LOADING</p>
          </div>
        </div>
      `;
    }

  function hideInitialLoading() {
    const loader = document.querySelector('.initial-loader');
    if (loader) loader.remove();
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

      const hasOnlyNavOptions =
        data.options &&
        data.options.length > 0 &&
        data.options.every(o => o.action === 'nav');

      appendMessage('bot', data.response, hasOnlyNavOptions ? 'wide-message' : '');
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
      appendMessage('bot', data.response, 'wide-message');

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
