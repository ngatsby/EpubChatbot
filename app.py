import streamlit as st
import tempfile
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from ebooklib import epub
from bs4 import BeautifulSoup

# --- 초기 설정 ---
GEMINI_API_KEY = st.secrets.get("Key")
if not GEMINI_API_KEY:
    st.error("Gemini API key not found. Please add 'Key' to your secrets.")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)

@st.cache_resource
def load_gemini_model():
    return genai.GenerativeModel("models/gemini-1.5-flash")

@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_gemini_model()
embedder = load_embedder()

# --- 함수들 ---

def extract_epub_chapters(epub_path):
    book = epub.read_epub(epub_path)
    titles, chapters = [], []

    # TOC에서 (title, href) 추출
    def flatten_toc(toc):
        result = []
        for item in toc:
            if isinstance(item, epub.Link):
                result.append((item.title, item.href))
            elif isinstance(item, tuple) and len(item) == 2:
                # (section_title, [subitems])
                # section_title = item[0]  # 필요시 사용
                result.extend(flatten_toc(item[1]))
            elif isinstance(item, list):
                result.extend(flatten_toc(item))
        return result

    toc_entries = flatten_toc(book.toc)

    for title, href in toc_entries:
        href_clean = href.split('#')[0]
        item = book.get_item_with_href(href_clean)
        if item is not None:
            soup = BeautifulSoup(item.get_content(), "html.parser")
            text = soup.get_text(separator="\n", strip=True)
            if len(text.strip()) > 100:  # 내용 없는 챕터는 제외
                titles.append(title.strip())
                chapters.append(text)
    return titles, chapters

def create_embeddings(texts):
    return embedder.encode(texts)

def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def ask_gemini(prompt_text):
    try:
        response = model.generate_content(prompt_text)
        return response.text.strip()
    except Exception as e:
        return f"❌ Gemini 응답 오류: {e}"

# --- 앱 UI ---

st.set_page_config(page_title="📚 ePub 챗봇", layout="wide")
st.title("📖 ePub 챕터 요약 & 챗봇")

uploaded_file = st.file_uploader("📤 ePub 파일 업로드", type="epub")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".epub") as tmp_file:
        tmp_file.write(uploaded_file.read())
        epub_path = tmp_file.name

    with st.spinner("📚 ePub 파일 처리 중..."):
        titles, chapters = extract_epub_chapters(epub_path)

    if not titles:
        st.error("❌ 챕터를 추출할 수 없습니다. ePub 구조를 확인해주세요.")
    else:
        st.success(f"✅ {len(titles)}개의 챕터를 추출했습니다.")
        chapter_idx = st.selectbox("🔍 요약할 챕터를 선택하세요:", range(len(titles)), format_func=lambda i: titles[i])

        selected_text = chapters[chapter_idx]

        with st.spinner("🧠 임베딩 및 요약 중..."):
            selected_embedding = create_embeddings([selected_text])
            index = build_faiss_index(np.array(selected_embedding))

            # 요약 요청
            summary_prompt_ko = f"다음 글을 한국어로 요약해줘:\n\n{selected_text[:4000]}"
            summary_prompt_en = f"Summarize the following text in English:\n\n{selected_text[:4000]}"

            summary_ko = ask_gemini(summary_prompt_ko)
            summary_en = ask_gemini(summary_prompt_en)

        st.subheader("📝 요약")
        st.markdown("**🇰🇷 한국어 요약:**")
        st.write(summary_ko)
        st.markdown("**🇺🇸 English Summary:**")
        st.write(summary_en)

        st.divider()
        st.subheader("💬 챕터 기반 질의응답")

        question = st.text_input("질문을 입력하세요 (선택한 챕터 기준)")
        if question:
            question_embedding = embedder.encode([question])
            D, I = index.search(np.array(question_embedding), k=1)
            context = selected_text
            prompt = f"""
다음 글을 참고하여 질문에 답해줘.

---

{context}

---

질문: {question}
"""
            answer = ask_gemini(prompt)
            st.markdown("**🤖 답변:**")
            st.write(answer)

        st.divider()
        st.subheader("🌐 전체 문서 기반 질의응답")

        global_question = st.text_input("전체 문서에 대해 질문하세요")
        if global_question:
            with st.spinner("전체 문서에서 답변 중..."):
                full_text = "\n".join(chapters)
                prompt = f"""
다음 ePub 전체 내용을 바탕으로 질문에 답하세요.

---

{full_text[:8000]}

---

질문: {global_question}
"""
                global_answer = ask_gemini(prompt)
                st.markdown("**🌍 전체 문서 응답:**")
                st.write(global_answer)

    try:
        os.remove(epub_path)
    except:
        pass
