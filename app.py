import streamlit as st

# 반드시 첫 줄에 위치!
st.set_page_config(page_title="📚 ePub 챗봇", layout="wide")

import tempfile
import os
from ebooklib import epub
from bs4 import BeautifulSoup
import google.generativeai as genai

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
    titles, chapters = [], []

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
            if len(text.strip()) > 100:
                titles.append(title.strip())
                chapters.append(text)
    return titles, chapters

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
        titles, chapters = extract_epub_chapters(epub_path)

    if not titles:
        st.error("❌ 챕터를 추출할 수 없습니다. ePub 구조를 확인해주세요.")
    else:
        st.success(f"✅ {len(titles)}개의 챕터를 추출했습니다.")
        chapter_idx = st.selectbox("🔍 요약할 챕터를 선택하세요:", range(len(titles)), format_func=lambda i: titles[i])

        # 반드시 선택한 챕터의 본문만 사용!
        selected_title = titles[chapter_idx]
        selected_text = chapters[chapter_idx]

        st.markdown(f"### 📄 선택한 챕터: {selected_title}")

        with st.spinner("🧠 요약 중..."):
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
