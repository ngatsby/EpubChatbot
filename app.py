import streamlit as st
import tempfile
import os
from ebooklib import epub
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import google.generativeai as genai

# --- Configuration ---
GEMINI_API_KEY = st.secrets.get("Key")
if not GEMINI_API_KEY:
    st.error("Gemini API key not found in Streamlit secrets. Please add 'Key' to your secrets.")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)

@st.cache_resource
def load_gemini_model():
    return genai.GenerativeModel("models/gemini-1.5-flash")

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_gemini_model()
embedder = load_embedding_model()

# --- Helper Functions ---
def extract_epub_chapters(epub_path):
    book = epub.read_epub(epub_path)
    chapters = []
    titles = []
    for item in book.get_items():
        if item.get_type() == epub.ITEM_DOCUMENT:
            soup = BeautifulSoup(item.get_content(), 'html.parser')
            text = soup.get_text(separator=' ', strip=True)
            title = soup.title.string if soup.title else f"Chapter {len(chapters)+1}"
            chapters.append(text)
            titles.append(title)
    return titles, chapters

def create_embeddings(texts):
    return embedder.encode(texts)

def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def ask_gemini(question, context):
    prompt = f"""
    다음 내용을 참고하여 질문에 답변해 주세요.
    ---
    {context}
    ---
    질문: {question}
    """
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Gemini 오류: {e}")
        return "응답 생성에 실패했습니다."

def summarize_chapter(text):
    kr_summary = ask_gemini("다음 내용을 한국어로 요약해 주세요:", text)
    en_summary = ask_gemini("Summarize the following content in English:", text)
    return kr_summary, en_summary

# --- Streamlit UI ---
st.title("📘 ePub 책/잡지 요약 및 챗봇")

if "epub_uploaded" not in st.session_state:
    st.session_state.epub_uploaded = False
if "chapters" not in st.session_state:
    st.session_state.chapters = []
if "titles" not in st.session_state:
    st.session_state.titles = []
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None

uploaded_file = st.file_uploader("ePub 파일을 업로드하세요", type="epub")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".epub") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    with st.spinner("ePub을 처리 중입니다..."):
        titles, chapters = extract_epub_chapters(tmp_path)
        st.session_state.titles = titles
        st.session_state.chapters = chapters
        st.session_state.embeddings = create_embeddings(chapters)
        st.session_state.faiss_index = build_faiss_index(np.array(st.session_state.embeddings))
        st.session_state.epub_uploaded = True
        st.success("ePub 처리 완료!")

    os.remove(tmp_path)

if st.session_state.epub_uploaded:
    st.subheader("📖 챕터 선택")
    chapter_options = {f"{i+1}. {title[:80]}": i for i, title in enumerate(st.session_state.titles)}
    selected_chapter_key = st.selectbox("요약할 챕터를 선택하세요", list(chapter_options.keys()))
    selected_idx = chapter_options[selected_chapter_key]

    selected_text = st.session_state.chapters[selected_idx]

    if st.button("이 챕터 요약하기"):
        with st.spinner("요약 중입니다..."):
            kr_summary, en_summary = summarize_chapter(selected_text)
            st.markdown("### 📌 한국어 요약")
            st.write(kr_summary)
            st.markdown("### 📌 English Summary")
            st.write(en_summary)

    st.markdown("---")
    st.subheader("💬 챕터 관련 질문하기")
    question = st.text_input("이 챕터에 대해 궁금한 점을 입력하세요")
    if question:
        answer = ask_gemini(question, selected_text)
        st.markdown("#### 🧠 Gemini의 답변")
        st.write(answer)

    st.markdown("---")
    st.subheader("🌐 전체 문서에 대해 질문하기")
    global_question = st.text_input("전체 문서 내용 기반 질문")
    if global_question:
        question_embedding = embedder.encode([global_question])
        D, I = st.session_state.faiss_index.search(np.array(question_embedding), k=5)
        retrieved_chunks = [st.session_state.chapters[i] for i in I[0] if i < len(st.session_state.chapters)]
        context = "\n".join(retrieved_chunks)
        answer = ask_gemini(global_question, context)
        st.markdown("#### 🧠 Gemini의 전체 문서 기반 답변")
        st.write(answer)
