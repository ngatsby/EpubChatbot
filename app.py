import streamlit as st

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
    chapters = []

    def flatten_toc(toc):
        result = []
        for item in toc:
            if isinstance(item, epub.Link):
                result.append((item.title.strip(), item.href))
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
        if item is None:
            continue
        soup = BeautifulSoup(item.get_content(), "html.parser")
        # 헤더 태그(h1~h3) 기준으로 챕터 분리
        headers = soup.find_all(['h1', 'h2', 'h3'])
        if not headers:
            # 헤더가 없으면 전체 텍스트
            text = soup.get_text(separator="\n", strip=True)
            if len(text.strip()) > 200:
                chapters.append({"title": title, "text": text.strip()})
            continue

        # 각 헤더별로 본문 분리
        for idx, header in enumerate(headers):
            chap_title = header.get_text(separator=" ", strip=True)
            if chap_title != title:
                continue
            # 본문 추출: 현재 헤더부터 다음 헤더 전까지
            content = []
            for sibling in header.next_siblings:
                if sibling.name in ['h1', 'h2', 'h3']:
                    break
                if isinstance(sibling, str):
                    content.append(sibling.strip())
                else:
                    content.append(sibling.get_text(separator="\n", strip=True))
            text = "\n".join([t for t in content if t])
            if len(text.strip()) > 100:
                chapters.append({"title": title, "text": text.strip()})
            break  # 같은 제목이 여러 번 등장할 경우 첫 번째만 사용

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

        st.markdown(f"#### 📄 선택한 챕터: {selected_title}")

        with st.spinner("🧠 요약 중..."):
            summary_prompt_ko = (
                f"아래 글은 '{selected_title}'라는 챕터의 전체 본문이야. "
                f"이 글만 참고해서 한국어로 5줄 이내로 요약해줘.\n\n"
                f"{selected_text[:4000]}"
            )
            summary_prompt_en = (
                f"This is the full text of the chapter titled '{selected_title}'. "
                f"Summarize ONLY this text in English in less than 5 sentences.\n\n"
                f"{selected_text[:4000]}"
            )

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
아래는 '{selected_title}'라는 챕터의 전체 본문이야. 이 본문만 참고해서 질문에 답해줘.

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
                full_text = "\n".join([c["text"] for c in chapters])
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
