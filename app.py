import streamlit as st

st.set_page_config(page_title="📚 ePub 챗봇", layout="wide")

import tempfile
import os
from ebooklib import epub
from bs4 import BeautifulSoup
import google.generativeai as genai
import textwrap

# --- 초기 설정 ---
GEMINI_API_KEY = st.secrets.get("Key")
if not GEMINI_API_KEY:
    st.error("Gemini API key not found. Please add 'Key' to your secrets.")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)

@st.cache_resource
def load_gemini_model():
    return genai.GenerativeModel("models/gemini-1.5-flash")

model = load_gemini_model()

# --- 함수들 ---

def extract_epub_chapters(epub_path):
    book = epub.read_epub(epub_path)
    chapters = []

    def flatten_toc(toc):
        result = []
        for item in toc:
            if isinstance(item, epub.Link):
                result.append((item.title, item.href))
            elif isinstance(item, tuple) and len(item) == 2:
                result.extend(flatten_toc(item[1]))
            elif isinstance(item, list):
                result.extend(flatten_toc(item))
        return result

    toc_entries = flatten_toc(book.toc)

    skip_keywords = ['표지', '차례', '목차', '저작권', '판권', 'prologue', 'contents', 'copyright', 'cover']

    for title, href in toc_entries:
        if any(k.lower() in title.lower() for k in skip_keywords):
            continue
        href_clean = href.split('#')[0]
        item = book.get_item_with_href(href_clean)
        if item is not None:
            soup = BeautifulSoup(item.get_content(), "html.parser")
            text = soup.get_text(separator="\n", strip=True)
            if len(text.strip()) > 200:
                chapters.append({"title": title.strip(), "text": text.strip()})
    return chapters

def ask_gemini(prompt_text):
    try:
        response = model.generate_content(prompt_text)
        return response.text.strip()
    except Exception as e:
        return f"❌ Gemini 응답 오류: {e}"

# --- 앱 UI ---

st.title("📖 ePub 챕터 요약 & 챗봇")

uploaded_file = st.file_uploader("📤 ePub 파일 업로드", type="epub")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".epub") as tmp_file:
        tmp_file.write(uploaded_file.read())
        epub_path = tmp_file.name

    with st.spinner("📚 ePub 파일 처리 중..."):
        chapters = extract_epub_chapters(epub_path)

    if not chapters:
        st.error("❌ 챕터를 추출할 수 없습니다. ePub 구조를 확인해주세요.")
    else:
        st.success(f"✅ {len(chapters)}개의 챕터를 추출했습니다.")
        chapter_titles = [c["title"] for c in chapters]
        chapter_idx = st.selectbox("🔍 요약할 챕터를 선택하세요:", range(len(chapters)), format_func=lambda i: chapter_titles[i])

        selected_chapter = chapters[chapter_idx]
        selected_title = selected_chapter["title"]
        selected_text = selected_chapter["text"]

        # 본문 미리보기 제거, 제목만 표시
        st.markdown(f"#### 📄 선택한 챕터: {selected_title}")

        with st.spinner("🧠 요약 중..."):
            summary_prompt_ko = (
                f"아래 글은 '{selected_title}'라는 챕터의 전체 본문이야. "
                f"이 글만 참고해서 한국어로 5줄 이내로 요약해줘.\n\n"
                f"{textwrap.shorten(selected
