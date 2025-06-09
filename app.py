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

# --- Load Models ---
@st.cache_resource(show_spinner=False)
def load_gemini_model():
    return genai.GenerativeModel("models/gemini-1.5-flash")

@st.cache_resource(show_spinner=False)
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_gemini_model()
embedder = load_embedding_model()

def extract_epub_chapters(epub_path):
    book = epub.read_epub(epub_path)

    # MIME 타입으로 문서만 필터링 (ITEM_DOCUMENT 대체)
    items = [item for item in book.items if item.media_type == "application/xhtml+xml"]

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
                # href에 #이 붙을 수 있으므로 분리
                if item.file_name == nav_point.href.split("#")[0]:
                    chapters.append(extract_text_from_item(item))
                    break
        elif isinstance(nav_point, (list, tuple)):
            for sub_point in nav_point:
                parse_nav(sub_point)

    parse_nav(toc)

    # 만약 toc가 너무 간단해서 빠진 챕터가 있을 경우, 모든 문서 추출 (옵션)
    if len(titles) == 0 or len(chapters) == 0:
        # 제목을 파일명으로 대체
        for item in items:
            titles.append(os.path.basename(item.file_name))
            chapters.append(extract_text_from_item(item))

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
    prompt = f"""
다음 내용으로 질문에 답해주세요.
---
{context}
---
질문: {question}
"""
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Error generating response from Gemini: {e}")
        return "죄송합니다. 답변을 생성하지 못했습니다."

# --- Streamlit UI ---
st.title("📚 ePub 챗봇 - 챕터 선택 후 요약 및 질의응답")

if "chapters" not in st.session_state:
    st.session_state.chapters = []
if "titles" not in st.session_state:
    st.session_state.titles = []
if "index" not in st.session_state:
    st.session_state.index = None
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "selected_chapter_idx" not in st.session_state:
    st.session_state.selected_chapter_idx = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

uploaded_file = st.file_uploader("ePub 파일 업로드", type=["epub"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".epub") as tmp_file:
        tmp_file.write(uploaded_file.read())
        epub_path = tmp_file.name

    with st.spinner("ePub에서 챕터 추출 중..."):
        titles, chapters = extract_epub_chapters(epub_path)
        st.session_state.titles = titles
        st.session_state.chapters = chapters

    try:
        os.remove(epub_path)
    except Exception:
        pass

if st.session_state.titles and st.session_state.chapters:
    st.subheader("목차")
    selected_idx = st.selectbox("챕터를 선택하세요", options=list(range(len(st.session_state.titles))),
                                format_func=lambda i: st.session_state.titles[i])

    if st.button("선택한 챕터 임베딩 및 요약 생성"):
        st.session_state.selected_chapter_idx = selected_idx
        chapter_text = st.session_state.chapters[selected_idx]

        # 임베딩 생성
        embeddings = create_embeddings([chapter_text])
        st.session_state.embeddings = embeddings
        st.session_state.index = build_faiss_index(embeddings)

        # 한국어 요약
        with st.spinner("한국어 요약 생성 중..."):
            summary_ko = ask_gemini(
                "아래 내용을 한국어로 간략히 요약해줘.",
                chapter_text
            )

        # 영어 요약
        with st.spinner("영어 요약 생성 중..."):
            summary_en = ask_gemini(
                "Please summarize the following content briefly in English.",
                chapter_text
            )

        st.markdown("### 선택한 챕터 요약")
        st.markdown("**한국어 요약:**")
        st.write(summary_ko)
        st.markdown("**English Summary:**")
        st.write(summary_en)

    if st.session_state.index is not None:
        st.subheader("선택한 챕터에 대해 질문하세요")
        user_question = st.text_input("질문 입력")

        if user_question:
            with st.spinner("답변 생성 중..."):
                question_embedding = embedder.encode([user_question])
                D, I = st.session_state.index.search(np.array(question_embedding), k=3)
                matched_texts = [st.session_state.chapters[st.session_state.selected_chapter_idx]]  # 단일 챕터에 한정

                context = "\n".join(matched_texts)
                answer = ask_gemini(user_question, context)
                st.markdown("#### Gemini 답변")
                st.write(answer)

    st.markdown("---")
    st.subheader("전체 문서에 대해 질문하기")
    all_text = "\n".join(st.session_state.chapters)
    all_embeddings = create_embeddings(st.session_state.chapters)
    all_index = build_faiss_index(all_embeddings)

    all_question = st.text_input("전체 문서 질문 입력", key="all_doc_question")

    if all_question:
        with st.spinner("전체 문서 답변 생성 중..."):
            question_emb = embedder.encode([all_question])
            D, I = all_index.search(np.array(question_emb), k=3)
            matched_all_texts = [st.session_state.chapters[i] for i in I[0] if i < len(st.session_state.chapters)]
            context_all = "\n".join(matched_all_texts)
            answer_all = ask_gemini(all_question, context_all)
            st.markdown("#### 전체 문서 답변")
            st.write(answer_all)
