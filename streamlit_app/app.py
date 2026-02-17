import os
import requests
import streamlit as st

st.set_page_config(page_title="社内チャットボット", layout="centered")

# ==========
# 設定（Streamlit Secrets → なければ環境変数）
# ==========
BASE_URL = None
if "CHAT_BASE_URL" in st.secrets:
    BASE_URL = st.secrets["CHAT_BASE_URL"]
else:
    BASE_URL = os.getenv("CHAT_BASE_URL", "")

BASE_URL = (BASE_URL or "").rstrip("/")

if not BASE_URL:
    st.error("CHAT_BASE_URL が未設定です。Manage app → Settings → Secrets に設定してください。")
    st.stop()

INIT_URL = f"{BASE_URL}/init"
SCENARIO_SELECT_URL = f"{BASE_URL}/scenario/select"
CHAT_URL = f"{BASE_URL}/chat"

# ==========
# state 初期化
# ==========
if "messages" not in st.session_state:
    st.session_state.messages = []  # [{"role":"bot/user","text":"..."}]

if "scenario_path" not in st.session_state:
    st.session_state.scenario_path = ["root"]

if "scenario_options" not in st.session_state:
    st.session_state.scenario_options = []

if "initialized" not in st.session_state:
    st.session_state.initialized = False

USER_ID = "anonymous_user"

# ==========
# 表示関数
# ==========
def bot_say(text: str):
    st.session_state.messages.append({"role": "bot", "text": text})

def user_say(text: str):
    st.session_state.messages.append({"role": "user", "text": text})

def render_chat():
    for m in st.session_state.messages:
        if m["role"] == "bot":
            st.chat_message("assistant").write(m["text"])
        else:
            st.chat_message("user").write(m["text"])

# ==========
# API呼び出し（エラー詳細を表示）
# ==========
def call_json(method: str, url: str, payload=None, timeout=30):
    try:
        if method == "GET":
            r = requests.get(url, timeout=timeout)
        else:
            r = requests.post(url, json=payload, timeout=timeout)

        # 失敗時に本文を表示できるようにする
        if not r.ok:
            return {"__error__": True, "status": r.status_code, "text": r.text}

        return r.json()
    except Exception as e:
        return {"__error__": True, "status": "EXCEPTION", "text": str(e)}

# ==========
# 初期化（/init）
# ==========
def init_scenario():
    data = call_json("GET", INIT_URL)

    if data.get("__error__"):
        bot_say("初期化に失敗しました。")
        st.error(f"INITエラー: {data.get('status')} / {data.get('text')}")
        return

    bot_say(data.get("response", ""))
    ui = data.get("ui") or {}
    if ui.get("path"):
        st.session_state.scenario_path = ui["path"]

    st.session_state.scenario_options = data.get("options", [])
    st.session_state.initialized = True

# ==========
# シナリオ選択（/scenario/select）
# ==========
def scenario_select(node_id: str, label: str):
    user_say(label)

    payload = {
        "node_id": node_id,
        "path": st.session_state.scenario_path,
        "user_id": USER_ID,
    }
    data = call_json("POST", SCENARIO_SELECT_URL, payload)

    if data.get("__error__"):
        bot_say("シナリオ処理でエラーが発生しました。")
        st.error(f"SCENARIOエラー: {data.get('status')} / {data.get('text')}")
        return

    bot_say(data.get("response", ""))
    ui = data.get("ui") or {}
    if ui.get("path"):
        st.session_state.scenario_path = ui["path"]

    st.session_state.scenario_options = data.get("options", [])

# ==========
# 自由入力（/chat）
# ==========
def free_chat(text: str):
    user_say(text)

    payload = {"message": text, "user_id": USER_ID}
    data = call_json("POST", CHAT_URL, payload)

    if data.get("__error__"):
        bot_say("自由入力でエラーが発生しました。")
        st.error(f"CHATエラー: {data.get('status')} / {data.get('text')}")
        return

    bot_say(data.get("response", ""))

# ==========
# UI
# ==========
st.title("社内チャットボット")

# 初期化ボタン（初回のみ自動実行でもOK）
if not st.session_state.initialized:
    init_scenario()

render_chat()

# ---- シナリオボタン群
opts = st.session_state.scenario_options or []
if opts:
    st.caption("カテゴリ選択（シナリオ）")
    cols = st.columns(3)
    for i, opt in enumerate(opts):
        col = cols[i % 3]
        if col.button(opt.get("label", opt.get("id", "")), key=f"opt_{i}_{opt.get('id')}"):
            scenario_select(opt.get("id"), opt.get("label", opt.get("id")))

st.divider()

# ---- 自由入力
text = st.chat_input("自由入力もできます（例：PPPoEとは？）")
if text:
    free_chat(text)
    st.rerun()
