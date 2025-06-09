import streamlit as st
import tempfile
import os
from ebooklib import epub
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import google.generativeai as genai

# --- 설정 ---
GEMINI_API_KEY = st.secrets.get("Key")
if not GEMINI_API_KEY:
    st.error("Gemini API key not found in Streamlit secrets.")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)

# --- 모델 불러오기 ---
@st.cache_resource
def load_gemini_model():
    return genai.GenerativeModel("models/gemini-1.5-flash")

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_gemini_model()
embedder = load_embedding_model()

# --- 함수: 챕터 추출 ---
def extract_epub_chapters(epub_path):
    book = epub.read_epub(epub_path)
    items = list(book.get_items_of_type(epub.ITEM_DOCUMENT))

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
                if item.file_name == nav_point.href.split("#")[0]:
                    chapters.append(extract_text_from_item(item))
                    break
        elif isinstance(nav_point, (list, tuple)):
            for sub_point in nav_point:
                parse_nav(sub_point)

    parse_nav(toc)
    return titles, chapters

# --- 함수: 임베딩 및 질의 ---
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

# --- UI ---
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
        st.success(f"{len(titles)}개 챕터를 찾았습니다.")
        selected_title = st.selectbox("🔍 챕터를 선택하세요", titles)

        if selected_title:
            chapter_idx = titles.index(selected_title)
            selected_text = chapters[chapter_idx]

            st.subheader("📝 한국어 요약")
            with st.spinner("요약 중..."):
                summary_ko = ask_gemini("다음 글을 한국어로 요약해줘.", selected_text)
                st.write(summary_ko)

            st.subheader("📝 English Summary")
            with st.spinner("Summarizing in English..."):
                summary_en = ask_gemini("Summarize the following content in English.", selected_text)
                st.write(summary_en)

            # --- 챕터 Q&A ---
            st.markdown("---")
            st.subheader("💬 이 챕터에 대해 질문하기")
            chapter_question = st.text_input("질문을 입력하세요 (챕터 기준)", key="chapter_q")

            if chapter_question:
                with st.spinner("답변 생성 중..."):
                    embeddings = create_embeddings([selected_text])
                    index = build_faiss_index(np.array(embeddings))
                    D, I = index.search(create_embeddings([chapter_question]), k=1)
                    context = selected_text
                    answer = ask_gemini(chapter_question, context)
                    st.markdown("##### 🤖 Gemini의 답변:")
                    st.write(answer)

        # --- 전체 문서 Q&A ---
        st.markdown("---")
        st.subheader("📖 전체 문서에 대해 질문하기")
        full_question = st.text_input("질문을 입력하세요 (전체 문서 기준)", key="full_q")

        if full_question:
            with st.spinner("전체 문서 임베딩 중..."):
                embeddings_all = create_embeddings(chapters)
                index_all = build_faiss_index(np.array(embeddings_all))
                D, I = index_all.search(create_embeddings([full_question]), k=3)
                context = "\n\n".join([chapters[i] for i in I[0]])
                answer = ask_gemini(full_question, context)
                st.markdown("##### 🤖 Gemini의 답변:")
                st.write(answer)
