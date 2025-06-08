import streamlit as st
import tempfile
import os
from ebooklib import epub, ITEM_DOCUMENT
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import google.generativeai as genai

# --- Gemini API Key Configuration ---
GEMINI_API_KEY = st.secrets.get("Key")
if not GEMINI_API_KEY:
    st.error("Gemini API key not found in Streamlit secrets. Please add 'Key' to your secrets.")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)

# --- Load models ---
@st.cache_resource
def load_model():
    return genai.GenerativeModel("models/gemini-1.5-flash")

@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()
embedder = load_embedder()

# --- Helper functions ---
def extract_epub_chapters(epub_path):
    book = epub.read_epub(epub_path)
    chapters = []
    titles = []
    for item in book.get_items():
        if item.get_type() == ITEM_DOCUMENT:
            soup = BeautifulSoup(item.get_content(), "html.parser")
            text = soup.get_text()
            title_tag = soup.find(['h1', 'h2', 'title'])
            title = title_tag.get_text().strip() if title_tag else f"Chapter {len(chapters) + 1}"
            chapters.append(text)
            titles.append(title)
    return titles, chapters

def create_embeddings(texts):
    return embedder.encode(texts)

def build_index(embeddings):
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index

def ask_gemini(question, context):
    prompt = f"""
    다음 글을 참고하여 질문에 답해주세요.
    ---
    {context}
    ---
    질문: {question}
    """
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"[Error] Gemini 응답 실패: {e}"

# --- Streamlit UI ---
st.title("📚 ePub 챗봇: 챕터 요약 + 질의응답")

uploaded_file = st.file_uploader("ePub 파일을 업로드하세요", type="epub")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".epub") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    with st.spinner("ePub을 처리 중입니다..."):
        titles, chapters = extract_epub_chapters(tmp_path)
        os.remove(tmp_path)

    if not chapters:
        st.error("챕터를 찾을 수 없습니다.")
        st.stop()

    selected = st.selectbox("챕터를 선택하세요", titles)
    selected_index = titles.index(selected)
    selected_text = chapters[selected_index]

    if st.button("챕터 요약하기"):
        with st.spinner("Gemini가 한국어 요약 중..."):
            ko_summary = ask_gemini("다음 내용을 한국어로 요약해줘.", selected_text)
            st.subheader("🇰🇷 한국어 요약")
            st.write(ko_summary)

        with st.spinner("Gemini가 영어 요약 중..."):
            en_summary = ask_gemini("Summarize the following content in English.", selected_text)
            st.subheader("🇺🇸 English Summary")
            st.write(en_summary)

    if st.checkbox("선택한 챕터에 질문하기"):
        query = st.text_input("질문을 입력하세요:")
        if query:
            with st.spinner("답변 생성 중..."):
                chapter_embedding = create_embeddings([selected_text])
                index = build_index(np.array(chapter_embedding))
                query_embedding = embedder.encode([query])
                D, I = index.search(np.array(query_embedding), k=1)
                matched_text = selected_text  # 전체 챕터 하나이므로 그대로 사용
                answer = ask_gemini(query, matched_text)
                st.markdown("#### 🤖 답변")
                st.write(answer)

    if st.checkbox("전체 문서에 질문하기"):
        all_text = "\n".join(chapters)
        global_query = st.text_input("전체 문서에 대한 질문:", key="global")
        if global_query:
            with st.spinner("전체 문서에서 답변 생성 중..."):
                answer = ask_gemini(global_query, all_text)
                st.markdown("#### 🌐 전체 문서 답변")
                st.write(answer)
