import streamlit as st

st.set_page_config(page_title="📚 ePub 챗봇", layout="wide")

import tempfile
import os
from ebooklib import epub
from bs4 import BeautifulSoup
import google.generativeai as genai
from gtts import gTTS  # TTS 라이브러리 추가
import io

GEMINI_API_KEY = st.secrets.get("Key")
if not GEMINI_API_KEY:
    st.error("Gemini API key not found. Please add 'Key' to your secrets.")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)

@st.cache_resource
def load_gemini_model():
    return genai.GenerativeModel("models/gemini-1.5-flash")
model = load_gemini_model()

def extract_epub_chapters(epub_path):
    book = epub.read_epub(epub_path)
    chapters = []

    # TOC 기반 추출
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
        text = soup.get_text(separator="\n", strip=True)
        if len(text.strip()) > 50:
            chapters.append({"title": title, "text": text.strip()})

    return chapters

def ask_gemini(prompt_text):
    try:
        response = model.generate_content(prompt_text)
        return response.text.strip()
    except Exception as e:
        return f"❌ Gemini 응답 오류: {e}"

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
        # 한국어 요약 TTS 버튼
        if st.button("🔊 한국어 요약 듣기"):
            tts_ko = gTTS(summary_ko, lang='ko')
            mp3_fp = io.BytesIO()
            tts_ko.write_to_fp(mp3_fp)
            st.audio(mp3_fp.getvalue(), format="audio/mp3")

        st.markdown("**🇺🇸 English Summary:**")
        st.write(summary_en)
        # 영어 요약 TTS 버튼
        if st.button("🔊 영어 요약 듣기"):
            tts_en = gTTS(summary_en, lang='en')
            mp3_fp = io.BytesIO()
            tts_en.write_to_fp(mp3_fp)
            st.audio(mp3_fp.getvalue(), format="audio/mp3")

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
            # 질의응답 TTS 버튼
            if st.button("🔊 답변 듣기"):
                tts_answer = gTTS(answer, lang='ko')
                mp3_fp = io.BytesIO()
                tts_answer.write_to_fp(mp3_fp)
                st.audio(mp3_fp.getvalue(), format="audio/mp3")

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
                # 전체 문서 응답 TTS 버튼
                if st.button("🔊 전체 문서 응답 듣기"):
                    tts_global = gTTS(global_answer, lang='ko')
                    mp3_fp = io.BytesIO()
                    tts_global.write_to_fp(mp3_fp)
                    st.audio(mp3_fp.getvalue(), format="audio/mp3")

    try:
        os.remove(epub_path)
    except:
        pass
