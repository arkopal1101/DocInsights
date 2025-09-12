# frontend/streamlit_app.py

import streamlit as st
import requests

BASE_URL = "https://arkpal1101-docinsight.hf.space"
st.title("📄 DocLens — Your Document Assistant")

upload_flag = False
uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
if uploaded_files and st.button("Upload to Backend"):
    with st.spinner("Uploading PDFs..."):
        files = [("files", (f.name, f.getvalue(), f.type)) for f in uploaded_files]
        res = requests.post(f"{BASE_URL}/upload_pdfs", files=files)
        if res.status_code == 200:
            upload_flag = True
            st.success("✅ PDFs uploaded and processed!")
        else:
            st.error(f"Upload failed: {res.text}")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

question = st.text_input("Ask a question about the documents:")

if st.button("Ask") or question:
    with st.spinner("🤔 Thinking..."):
        res = requests.post(f"{BASE_URL}/ask", json={"question": question})
        if res.status_code == 200:
            result = res.json()
            st.session_state.chat_history.append({
                "question": question,
                "answer": result["answer"]
            })
            st.markdown("💡 Answer:")
            st.write(result['answer'])

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
        st.rerun()