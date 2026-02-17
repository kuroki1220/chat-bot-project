import streamlit as st
import requests

# Cloud Run のベースURL（末尾 /chat じゃなくてサービスのルート）
BASE_URL = st.secrets.get("BASE_URL", "https://chatbot-backend-140594287961.asia-northeast1.run.app")

INIT_URL = f"{BASE_URL}/init"
SELECT_URL = f"{BASE_URL}/scenario/select"
CHAT_URL = f"{BASE_URL}/chat"

st.set_page_config(page_title="社内チャットボット", layout="centered")
st.title("社内チャットボット")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "scenario_path" not in st.session_state:
    st.session_state.scenario_path = ["root"]
if "options" not in st.session_state:
    st.session_state.options = []

def bot_say(text):
    st.session_state.messages.append(("bot", text))

def user_say(text):
    st.session_state.messages.append(("user", text))

def init():
    r = requests.get(INIT_URL, timeout=30)
    r.raise_for_status()
    data = r.json()
    bot_say(data.get("response", ""))
    st.session_state.options = data.get("options", [])
    ui = data.get("ui") or {}
    if ui.get("path"):
        st.session_state.scenario_path = ui["path"]

def scenario_select(node_id, label):
    user_say(label)
    payload = {
        "node_id": node_id,
        "path": st.session_state.scenario_path,
        "user_id": "anonymous_user"
    }
    r = requests.post(SELECT_URL, json=payload, timeout=30)
    r.raise_for_status()
    data = r.json()
    bot_say(data.get("response", ""))
    st.session_state.options = data.get("options", [])
    ui = data.get("ui") or {}
    if ui.get("path"):
        st.session_state.scenario_path = ui["path"]

def free_chat(text):
    user_say(text)
    payload = {"message": text, "user_id": "anonymous_user"}
    r = requests.post(CHAT_URL, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()
    bot_say(data.get("response", ""))

# 初期化
if len(st.session_state.messages) == 0:
    try:
        init()
    except Exception as e:
        bot_say("初期化に失敗しました。バックエンドURLや公開設定を確認してください。")
        st.error(e)

# 吹き出し表示
for who, text in st.session_state.messages:
    with st.chat_message("assistant" if who == "bot" else "user"):
        st.write(text)

# シナリオボタン（大カテゴリ/中カテゴリなど）
if st.session_state.options:
    cols = st.columns(4)
    for i, opt in enumerate(st.session_state.options):
        with cols[i % 4]:
            if st.button(opt["label"], key=f"opt-{opt['id']}-{i}"):
                scenario_select(opt["id"], opt["label"])
                st.rerun()

# 自由入力
text = st.chat_input("質問を入力してください")
if text:
    free_chat(text)
    st.rerun()
