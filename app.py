import streamlit as st
import tempfile
import os
from ebooklib import epub
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import google.generativeai as genai

# --- API Key ---
GEMINI_API_KEY = st.secrets.get("Key")
if not GEMINI_API_KEY:
    st.error("Gemini API key not found. Please add 'Key' to your Streamlit secrets.")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)

# --- Load Models ---
@st.cache_resource
def load_gemini_model():
    return genai.GenerativeModel("models/gemini-1.5-flash")

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_gemini_model()
embedder = load_embedding_model()

# --- Utils ---
def extract_epub_chapters(epub_path):
    book = epub.read_epub(epub_path)
    titles = []
    chapters = []

    def extract_from_toc(toc_items):
        for item in toc_items:
            if isinstance(item, epub.Link):
                doc = book.get_item_with_href(item.href)
                if doc:
                    soup = BeautifulSoup(doc.get_body_content(), "html.parser")
                    text = soup.get_text(separator="\n").strip()
                    if text:
                        titles.append(item.title)
                        chapters.append(text)
            elif isinstance(item, (list, tuple)):
                extract_from_toc(item)

    extract_from_toc(book.toc)

    # fallback if no toc items
    if not chapters:
        for item in book.get_items_of_type(epub.ITEM_DOCUMENT):
            soup = BeautifulSoup(item.get_content(), "html.parser")
            text = soup.get_text(separator="\n").strip()
            if text:
                titles.append(item.get_name())
                chapters.append(text)

    return titles, chapters

def create_embeddings(texts):
    return embedder.encode(texts) if texts else np.array([])

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

# --- Streamlit UI ---
st.title("📘 ePub 챕터 기반 요약 & Gemini QnA")

uploaded_file = st.file_uploader("ePub 파일 업로드", type="epub")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".epub") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    with st.spinner("📖 ePub 파일 분석 중..."):
        titles, chapters = extract_epub_chapters(tmp_path)
        os.remove(tmp_path)

    if not chapters:
        st.error("본문을 추출할 수 없습니다.")
        st.stop()

    # 인덱스 보정
    min_len = min(len(titles), len(chapters))
    titles, chapters = titles[:min_len], chapters[:min_len]

    st.success(f"{len(chapters)}개의 챕터가 준비되었습니다.")
    chapter_idx = st.selectbox("📚 챕터를 선택하세요", list(range(min_len)), format_func=lambda i: titles[i])
    selected_text = chapters[chapter_idx]

    st.subheader("📝 챕터 한글 요약")
    st.write(ask_gemini("이 내용을 한국어로 요약해줘", selected_text))

    st.subheader("📝 Chapter Summary (English)")
    st.write(ask_gemini("Summarize this chapter in English.", selected_text))

    # 챕터 기반 QnA
    st.subheader("💬 챕터 질문")
    with st.spinner("📊 임베딩 처리 중..."):
        chapter_embedding = create_embeddings([selected_text])
        chapter_index = build_faiss_index(np.array(chapter_embedding))

    chapter_q = st.text_input("질문을 입력하세요", key="chapter_q")
    if chapter_q and chapter_index:
        q_emb = embedder.encode([chapter_q])
        D, I = chapter_index.search(np.array(q_emb), k=1)
        context = selected_text
        st.markdown("**🧠 Gemini의 답변:**")
        st.write(ask_gemini(chapter_q, context))

    # 전체 문서 기반 QnA
    st.divider()
    st.subheader("🌍 전체 문서 질문")

    all_embeddings = create_embeddings(chapters)
    all_index = build_faiss_index(np.array(all_embeddings))
    doc_q = st.text_input("전체 문서에 대한 질문을 입력하세요", key="doc_q")

    if doc_q and all_index:
        q_emb = embedder.encode([doc_q])
        D, I = all_index.search(np.array(q_emb), k=3)
        context = "\n".join([chapters[i] for i in I[0] if i < len(chapters)])
        st.markdown("**🧠 Gemini의 답변:**")
        st.write(ask_gemini(doc_q, context))
