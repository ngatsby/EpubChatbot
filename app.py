import streamlit as st
import tempfile
import os
from ebooklib import epub
from ebooklib import ITEM_DOCUMENT  # 🔧 수정: ITEM_DOCUMENT 직접 import
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import google.generativeai as genai

# --- API 키 설정 ---
GEMINI_API_KEY = st.secrets.get("Key")
if not GEMINI_API_KEY:
    st.error("Gemini API key not found in Streamlit secrets.")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)

# --- 모델 로드 ---
@st.cache_resource
def load_gemini_model():
    return genai.GenerativeModel("models/gemini-1.5-flash")

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_gemini_model()
embedder = load_embedding_model()

# --- ePub 챕터 추출 함수 ---
def extract_epub_chapters(epub_path):
    book = epub.read_epub(epub_path)
    items = list(book.get_items_of_type(ITEM_DOCUMENT))  # 🔧 여기서 오류 발생했음

    toc = book.toc
    chapters = []
    titles = []

    def extract_text_from_item(item):
        soup = BeautifulSoup(item.get_content(), "html.parser")
        return soup.get_text(separator="\n").strip()

    def parse_nav(nav_point):
        if isinstance(nav_point, epub.Link):
            titles.append(nav_point.title)
            for item in items:
                if item.file_name.split("#")[0] == nav_point.href.split("#")[0]:
                    chapters.append(extract_text_from_item(item))
                    break
        elif isinstance(nav_point, (list, tuple)):
            for sub_point in nav_point:
                parse_nav(sub_point)

    parse_nav(toc)
    return titles, chapters

# --- 기타 유틸리티 함수 ---
def create_embeddings(texts):
    return embedder.encode(texts)

def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def ask_gemini(question, context):
    prompt = f"""
    아래 내용을 참고하여 질문에 답하세요.

    문맥:
    {context}

    질문:
    {question}
    """
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Gemini 응답 오류: {e}"

# --- Streamlit 인터페이스 ---
st.title("📚 ePub 챗봇: 목차 기반 요약 및 질의응답")

uploaded_file = st.file_uploader("📤 ePub 파일 업로드", type="epub")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".epub") as tmp_file:
        tmp_file.write(uploaded_file.read())
        epub_path = tmp_file.name

    with st.spinner("ePub 파일 분석 중..."):
        titles, chapters = extract_epub_chapters(epub_path)
        os.remove(epub_path)

    if not titles:
        st.warning("목차를 찾을 수 없습니다.")
    else:
        st.success(f"{len(titles)}개 챕
