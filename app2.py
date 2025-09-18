import streamlit as st
import google.generativeai as genai
import ollama
import io
from PIL import Image
from ocr_utils import extract_text_from_image   # <-- OCR module

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Gemini + Ollama Chat", page_icon="🤖", layout="centered")

# Configure Gemini
genai.configure(api_key="AIzaSyBN_bOwZpV4qterCrRWiZMclqai6CkZKhQ")   # 🔑 replace with your key
GEMINI_MODEL = "gemini-1.5-flash"
OLLAMA_MODEL = "llama3.1:8b"

# ---------------- SESSION STATE ----------------
if "history" not in st.session_state: st.session_state.history = {}
if "chat_counter" not in st.session_state: st.session_state.chat_counter = 1
if "current_chat" not in st.session_state: st.session_state.current_chat = None
if "rename_chat" not in st.session_state: st.session_state.rename_chat = None
if "system_prompt" not in st.session_state: st.session_state.system_prompt = "You are a helpful assistant."
if "model_choice" not in st.session_state: st.session_state.model_choice = "Gemini API"

# Initialize first chat
if not st.session_state.history:
    cid = f"chat_{st.session_state.chat_counter}"
    st.session_state.chat_counter += 1
    st.session_state.current_chat = cid
    st.session_state.history[cid] = {"name": "Default Chat","messages":[{"role":"system","content":st.session_state.system_prompt}]}
elif st.session_state.current_chat is None:
    st.session_state.current_chat = list(st.session_state.history.keys())[0]

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("## 🤖 Choose Model")
    st.session_state.model_choice = st.radio("Select Model:", ["Gemini API", "Ollama Local"])

    if st.button("➕ New Chat", use_container_width=True):
        cid = f"chat_{st.session_state.chat_counter}"
        st.session_state.chat_counter += 1
        st.session_state.current_chat = cid
        st.session_state.history[cid] = {"name": "New Chat","messages":[{"role":"system","content":st.session_state.system_prompt}]}

    st.markdown("## 💾 Chat History")
    for cid, chat in st.session_state.history.items():
        cols = st.columns([0.8,0.2])
        if cols[0].button(chat["name"], key=f"chat_{cid}", use_container_width=True):
            st.session_state.current_chat = cid
        if cols[1].button("⋮", key=f"menu_{cid}"): st.session_state.rename_chat = cid
        if st.session_state.rename_chat == cid:
            new_name = st.text_input("Rename", chat["name"], key=f"rename_input_{cid}")
            if st.button("✅ Save", key=f"save_{cid}"):
                st.session_state.history[cid]["name"] = new_name or chat["name"]; st.session_state.rename_chat=None
            if st.button("🗑 Delete", key=f"delete_{cid}"):
                del st.session_state.history[cid]; st.session_state.rename_chat=None
                st.session_state.current_chat = list(st.session_state.history.keys())[0] if st.session_state.history else None
                st.rerun()

# ---------------- MAIN UI ----------------
st.title("💬 Chat with Gemini or Ollama")

if st.session_state.current_chat:
    chat = st.session_state.history[st.session_state.current_chat]

    # Display previous messages
    for msg in chat["messages"]:
        if msg["role"]!="system":
            with st.chat_message(msg["role"]): st.markdown(msg["content"])

    # ---------- OCR Upload ----------
    st.subheader("📷 Upload Image for OCR")
    uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        extracted_text = extract_text_from_image(image)
        if extracted_text:
            st.success("✅ OCR Extracted Text Added to Chat Context")
            chat["messages"].append({"role": "user", "content": extracted_text})

    # ---------- Chat Input ----------
    prompt = st.chat_input("Ask something...")
    if prompt:
        if chat["name"] in ["New Chat","Default Chat"]:
            chat["name"] = prompt[:40]+("..." if len(prompt)>40 else "")
        chat["messages"].append({"role":"user","content":prompt})
        with st.chat_message("user"): st.markdown(prompt)

        with st.chat_message("assistant"):
            placeholder, full_text = st.empty(), ""
            try:
                if st.session_state.model_choice == "Gemini API":
                    # ---- Gemini Response ----
                    model = genai.GenerativeModel(GEMINI_MODEL)
                    stream = model.generate_content(
                        [m["content"] for m in chat["messages"] if m["role"] in ("system","user")],
                        stream=True,
                    )
                    for chunk in stream:
                        if chunk.text:
                            full_text += chunk.text
                            placeholder.markdown(full_text)

                else:
                    # ---- Ollama Response ----
                    stream = ollama.chat(
                        model=OLLAMA_MODEL,
                        messages=chat["messages"],
                        stream=True
                    )
                    for chunk in stream:
                        if "message" in chunk and "content" in chunk["message"]:
                            c = chunk["message"]["content"]
                            full_text += c
                            placeholder.markdown(full_text)

            except Exception as e:
                full_text=f"❌ Error: {e}"
                placeholder.error(full_text)

            chat["messages"].append({"role":"assistant","content":full_text})

    # ---------- Download Chat ----------
    if len(chat["messages"])>1:
        txt="".join([f"{m['role'].upper()}: {m['content']}\n\n" for m in chat["messages"] if m["role"]!="system"])
        st.download_button("📤 Share Current Chat", data=io.BytesIO(txt.encode()), file_name=f"{chat['name'].replace(' ','_')}.txt", mime="text/plain")

else:
    st.info("Start a new chat from the sidebar ➕")
