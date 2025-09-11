# frontend/streamlit_app.py

import streamlit as st
import requests

BASE_URL = "http://localhost:8000"
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

question = st.text_input("Ask a question about the documents:")

if st.button("Ask") or question:
    with st.spinner("🤔 Thinking..."):
        res = requests.post(f"{BASE_URL}/ask", json={"question": question})
        if res.status_code == 200:
            result = res.json()
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
