# frontend/streamlit_app.py
import uuid

import streamlit as st
import requests

BASE_URL = "https://arkpal1101-docinsight.hf.space"
st.title("📄 DocInsights — Your Document Assistant")

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "last_answer" not in st.session_state:
    st.session_state.last_answer = ""
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "upload_flag" not in st.session_state:
    st.session_state.upload_flag = False

st.session_state.uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
if st.session_state.uploaded_files and st.button("Upload to Backend"):
    with st.spinner("Uploading PDFs..."):
        files = [("files", (f.name, f.getvalue(), f.type)) for f in st.session_state.uploaded_files]
        res = requests.post(f"{BASE_URL}/upload_pdfs", params={"session_id": st.session_state.session_id}, files=files)
        if res.status_code == 200:
            st.success("✅ PDFs uploaded and processed!")
            st.session_state.upload_flag = True
        else:
            st.error(f"Upload failed: {res.text}")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

question = st.text_input("Ask a question about the documents:")
go_label = "Go" if st.session_state.upload_flag else "🔒 Upload PDFs First"
button = st.button(go_label, disabled=not st.session_state.upload_flag)

if button and question:
    with st.spinner("🤔 Thinking..."):
        res = requests.post(f"{BASE_URL}/ask",
                            json={"session_id": st.session_state.session_id, "question": question})
        if res.status_code == 200:
            result = res.json()
            st.session_state.chat_history.append({
                "question": question,
                "answer": result["answer"]
            })
            st.session_state.last_answer = result['answer']
            if st.session_state.last_answer:
                st.markdown("💡 Answer:")
                st.write(st.session_state.last_answer)

            with st.expander("📚 Sources"):
                for src in result["sources"]:
                    source_name = src.get("source", "N/A")
                    short_name = (source_name[:100] + "...") if len(source_name) > 100 else source_name

                    st.markdown(f"**{short_name}** (page {src['page']})")
                    st.write(src["snippet"])
                    st.markdown("---")  # divider between sources

        else:
            st.error("Error getting answer from backend")

with st.sidebar:
    with st.expander("### 📝 Last 3 Questions"):
        for turn in st.session_state.chat_history[-3:]:
            st.write(f"Q: {turn['question']}")

# Then at the very end:
with st.sidebar:
    for i in range(38):
        st.write("")
    st.markdown("---")  # optional separator
    if st.button("🔄 Reset Session", key="reset_button"):
        st.session_state.clear()
        st.session_state["session_id"] = str(uuid.uuid4())
        st.rerun()
