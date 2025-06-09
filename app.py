import streamlit as st
import tempfile
import os
from ebooklib import epub
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import google.generativeai as genai

# --- API Key 설정 ---
GEMINI_API_KEY = st.secrets.get("Key")
if not GEMINI_API_KEY:
    st.error("Gemini API key not found. Please add 'Key' to your Streamlit secrets.")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)

# --- 모델 로딩 ---
@st.cache_resource
def load_gemini_model():
    return genai.GenerativeModel("models/gemini-1.5-flash")

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_gemini_model()
embedder = load_embedding_model()

# --- 함수 정의 ---
def extract_epub_chapters(epub_path):
    book = epub.read_epub(epub_path)
    titles = []
    chapters = []

    def extract_from_toc(toc_items):
        for item in toc_items:
            if isinstance(item, epub.Link):
                titles.append(item.title)
                doc = book.get_item_with_href(item.href)
                if doc:
                    soup = BeautifulSoup(doc.get_body_content(), "html.parser")
                    chapters.append(soup.get_text(separator="\n"))
            elif isinstance(item, (list, tuple)):
                extract_from_toc(item)

    extract_from_toc(book.toc)

    # 보완: 목차 정보 없을 때 추정으로 대체
    if not titles:
        for item in book.get_items_of_type(epub.ITEM_DOCUMENT):
            soup = BeautifulSoup(item.get_content(), "html.parser")
            text = soup.get_text(separator="\n")
            if any(kw in text.lower() for kw in ['목차', '차례', 'contents', 'table of contents', 'index']):
                titles.append(f"[추정목차] {item.get_name()}")
                chapters.append(text)

    return titles, chapters

def create_embeddings(texts):
    if not texts:
        return np.array([])
    return embedder.encode(texts)

def build_faiss_index(embeddings):
    if embeddings.size == 0:
        return None
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def ask_gemini(question, context):
    prompt = f"""다음 내용을 참고하여 질문에 답해주세요:
---
{context}
---
질문: {question}
"""
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"[Gemini 응답 오류] {e}"

# --- Streamlit UI 시작 ---
st.title("📘 ePub 챕터 기반 Gemini 요약 & QnA 챗봇")

uploaded_file = st.file_uploader("ePub 파일 업로드", type="epub")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".epub") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    with st.spinner("📖 ePub 파일 처리 중..."):
        titles, chapters = extract_epub_chapters(tmp_path)
        os.remove(tmp_path)

    if not titles:
        st.error("챕터(목차)를 추출할 수 없습니다.")
        st.stop()

    st.success(f"✅ {len(titles)}개의 챕터가 추출되었습니다.")
    chapter_idx = st.selectbox("📚 읽고 싶은 챕터를 선택하세요", list(range(len(titles))), format_func=lambda i: titles[i])
    selected_text = chapters[chapter_idx]

    if selected_text:
        st.subheader("📝 챕터 한글 요약")
        st.write(ask_gemini("이 내용을 한국어로 요약해줘", selected_text))

        st.subheader("📝 Chapter Summary (English)")
        st.write(ask_gemini("Summarize this chapter in English.", selected_text))

        with st.spinner("📊 임베딩 처리 중..."):
            embeddings = create_embeddings([selected_text])
            index = build_faiss_index(np.array(embeddings))

        st.subheader("💬 챕터에 대해 질문해보세요")
        chapter_question = st.text_input("질문을 입력하세요", key="chapter_q")
        if chapter_question and index:
            q_emb = embedder.encode([chapter_question])
            D, I = index.search(np.array(q_emb), k=1)
            context = selected_text
            st.markdown("**🧠 Gemini의 답변:**")
            st.write(ask_gemini(chapter_question, context))

        st.divider()
        st.subheader("🌍 전체 문서에 대해 질문하기")

        # 전체 벡터 저장 및 검색
        all_embeddings = create_embeddings(chapters)
        all_index = build_faiss_index(np.array(all_embeddings))

        doc_question = st.text_input("전체 문서에 대한 질문을 입력하세요", key="doc_q")
        if doc_question and all_index:
            q_emb = embedder.encode([doc_question])
            D, I = all_index.search(np.array(q_emb), k=3)
            context = "\n\n".join([chapters[i] for i in I[0]])
            st.markdown("**🧠 Gemini의 답변:**")
            st.write(ask_gemini(doc_question, context))
