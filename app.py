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
    """Gemini 모델을 로드합니다."""
    return genai.GenerativeModel("models/gemini-1.5-flash")

@st.cache_resource
def load_embedder():
    """임베딩 모델을 로드합니다."""
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_gemini_model()
embedder = load_embedder()

# --- 함수들 ---
def extract_epub_chapters(epub_path):
    """
    ePub 파일에서 챕터 제목과 내용을 추출합니다.
    ePub 목차(TOC), HTML <title> 태그, 그리고 HTML 헤딩 태그를 사용하여
    가장 정확한 챕터 제목을 찾습니다.
    """
    book = epub.read_epub(epub_path)
    titles, chapters = [], []
    
    # 챕터 파일 경로(href)와 해당 제목을 매핑하기 위한 딕셔너리
    toc_href_to_title = {}

    # ePub 목차 항목을 재귀적으로 처리하여 href와 제목 매핑을 만듭니다.
    def process_toc_entry(entry):
        if isinstance(entry, epub.EpubNavPoint):
            # EpubNavPoint는 목차의 특정 지점을 나타내며, 제목과 href를 가집니다.
            href_base = entry.href.split('#')[0] # 앵커(#) 부분 제거
            if entry.title:
                toc_href_to_title[href_base] = entry.title.strip()
            
            # 중첩된 목차 항목(자식)이 있다면 재귀적으로 처리합니다.
            if entry.children:
                for child in entry.children:
                    process_toc_entry(child)
        elif isinstance(entry, epub.EpubHtml):
            # EpubHtml 항목이 목차에 직접 포함될 수 있습니다.
            href_base = entry.href.split('#')[0]
            if hasattr(entry, 'title') and entry.title: # EpubHtml 객체에 제목 속성이 있는 경우
                toc_href_to_title[href_base] = entry.title.strip()
            # EpubHtml에도 subitems가 있을 수 있으므로 처리합니다.
            if hasattr(entry, 'subitems') and entry.subitems:
                 for sub_item in entry.subitems:
                    process_toc_entry(sub_item)
        elif isinstance(entry, tuple) and len(entry) == 2 and isinstance(entry[0], epub.EpubHtml):
            # 일부 ePub 목차는 (EpubHtml 객체, 제목 문자열) 형태의 튜플로 구성될 수 있습니다.
            href_base = entry[0].href.split('#')[0]
            if entry[1]: # 튜플의 두 번째 요소가 제목입니다.
                toc_href_to_title[href_base] = entry[1].strip()
            if hasattr(entry[0], 'subitems') and entry[0].subitems:
                 for sub_item in entry[0].subitems:
                    process_toc_entry(sub_item)

    # book.toc에 있는 모든 최상위 목차 항목부터 처리를 시작합니다.
    for item in book.toc:
        process_toc_entry(item)

    # 이제 모든 문서 항목을 반복하며 내용을 추출하고 가장 적절한 제목을 적용합니다.
    for item in book.get_items():
        if item.get_type() == epub.ITEM_DOCUMENT: # 실제 HTML 콘텐츠 파일인 경우
            soup = BeautifulSoup(item.get_content(), "html.parser")
            
            chapter_title = ""
            # 현재 항목의 파일 이름을 정규화합니다 (href와 유사).
            item_file_name_normalized = item.file_name.split('#')[0]

            # 1. 첫 번째 시도: 목차 맵에서 제목을 가져옵니다.
            if item_file_name_normalized in toc_href_to_title:
                chapter_title = toc_href_to_title[item_file_name_normalized]
            
            # 2. 두 번째 시도: HTML 콘텐츠 내의 <title> 태그를 확인합니다.
            if not chapter_title:
                title_tag = soup.find('title')
                if title_tag and title_tag.string:
                    chapter_title = title_tag.string.strip()
            
            # 3. 세 번째 시도: HTML 콘텐츠 내의 첫 번째 헤딩(h1-h6)을 제목으로 사용합니다.
            if not chapter_title:
                for h_level in range(1, 7): # h1부터 h6까지 반복
                    heading_tag = soup.find(f'h{h_level}')
                    if heading_tag and heading_tag.get_text(strip=True):
                        chapter_title = heading_tag.get_text(strip=True)
                        break # 제목을 찾았으면 더 이상 찾지 않습니다.
            
            # 4. 최종 대체: 제목을 찾지 못했다면 "Unnamed Chapter"를 할당합니다.
            if not chapter_title:
                chapter_title = f"Unnamed Chapter {len(titles)+1}"
            
            # 챕터의 전체 텍스트 내용을 추출합니다.
            text = soup.get_text(separator="\n", strip=True)
            
            # 내용이 충분한 챕터만 포함합니다 (예: 길이가 100자 이상).
            if len(text.strip()) > 100:
                titles.append(chapter_title)
                chapters.append(text)
                
    return titles, chapters

def create_embeddings(texts):
    """주어진 텍스트 목록에 대한 임베딩을 생성합니다."""
    return embedder.encode(texts)

def build_faiss_index(embeddings):
    """임베딩으로 FAISS 인덱스를 구축합니다."""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def ask_gemini(prompt_text):
    """Gemini 모델에 질문을 하고 응답을 반환합니다."""
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
    # 임시 파일에 업로드된 ePub 파일을 저장합니다.
    with tempfile.NamedTemporaryFile(delete=False, suffix=".epub") as tmp_file:
        tmp_file.write(uploaded_file.read())
        epub_path = tmp_file.name

    with st.spinner("📚 ePub 파일 처리 중..."):
        titles, chapters = extract_epub_chapters(epub_path)

    if not titles:
        st.error("❌ 챕터를 추출할 수 없습니다. ePub 구조를 확인해주세요.")
    else:
        st.success(f"✅ {len(titles)}개의 챕터를 추출했습니다.")
        # 챕터 선택을 위한 드롭다운 메뉴를 만듭니다.
        chapter_idx = st.selectbox("🔍 요약할 챕터를 선택하세요:", range(len(titles)), format_func=lambda i: titles[i])

        selected_text = chapters[chapter_idx]

        with st.spinner("🧠 임베딩 및 요약 중..."):
            selected_embedding = create_embeddings([selected_text])
            index = build_faiss_index(np.array(selected_embedding))

            # 한국어 및 영어 요약 요청
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
        # 임시 파일 정리
        os.remove(epub_path)
    except Exception as e:
        st.error(f"임시 파일 삭제 중 오류 발생: {e}")
