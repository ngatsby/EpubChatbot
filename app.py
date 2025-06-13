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

    def process_toc_item_recursively(toc_entry_item):
        """
        Recursively processes an ePub TOC entry to extract its href and title.
        Handles EpubNavPoint, EpubHtml, EpubSection, and common tuple structures.
        """
        href = None
        title = None
        children = []

        # 튜플 유형을 가장 먼저 처리하여 AttributeError 방지
        if isinstance(toc_entry_item, tuple):
            if len(toc_entry_item) >= 1:
                possible_item = toc_entry_item[0]
                if isinstance(possible_item, (epub.EpubHtml, epub.EpubNavPoint, epub.Section)):
                    href = getattr(possible_item, 'href', None)
                    if len(toc_entry_item) > 1:
                        # 튜플의 두 번째 요소가 일반적으로 제목입니다.
                        title = toc_entry_item[1] 
                    # 자식 항목도 함께 처리
                    children = getattr(possible_item, 'children', []) or getattr(possible_item, 'subitems', [])
                elif len(toc_entry_item) == 2 and isinstance(toc_entry_item[0], str) and isinstance(toc_entry_item[1], str):
                    # (href_string, title_string) 형태의 튜플 처리
                    href = toc_entry_item[0]
                    title = toc_entry_item[1]
        
        elif isinstance(toc_entry_item, epub.EpubNavPoint):
            # 표준 목차 내비게이션 포인트
            href = toc_entry_item.href
            title = toc_entry_item.title
            children = toc_entry_item.children
                
        elif isinstance(toc_entry_item, epub.EpubHtml):
            # EpubHtml 객체가 TOC에 직접 포함된 경우
            href = toc_entry_item.href
            title = getattr(toc_entry_item, 'title', None) # EpubHtml에도 title 속성이 있을 수 있음
            children = getattr(toc_entry_item, 'subitems', []) # EpubHtml도 하위 항목을 가질 수 있음
        
        elif isinstance(toc_entry_item, epub.Section):
            # EpubSection 객체가 TOC에 직접 포함된 경우 (중첩된 섹션용)
            title = toc_entry_item.title
            children = toc_entry_item.subitems # Section은 하위 항목을 가짐

        # href가 있는 경우에만 (정규화된 href, 제목) 쌍을 반환
        if href:
            normalized_href = href.split('#')[0]
            yield (normalized_href, title.strip() if title else "")

        # 자식 항목이 있다면 재귀적으로 처리
        for child in children:
            yield from process_toc_item_recursively(child)

    # book.toc를 반복하여 toc_href_to_title 맵을 채웁니다.
    for toc_entry in book.toc:
        for href, title in process_toc_item_recursively(toc_entry):
            # 제목이 있거나 새로운 href 항목인 경우에만 추가 (기존 제목이 더 정확할 수 있으므로 덮어쓰지 않음)
            if href not in toc_href_to_title or (title and not toc_href_to_title[href]):
                toc_href_to_title[href] = title

    # 이제 모든 문서 항목을 반복하며 내용을 추출하고 최적의 제목을 적용합니다.
    for item in book.get_items():
        if item.get_type() == epub.ITEM_DOCUMENT: # HTML 콘텐츠 파일인 경우
            soup = BeautifulSoup(item.get_content(), "html.parser")
            
            chapter_title = ""
            # TOC href와 일치하도록 항목의 파일 이름을 정규화
            item_file_name_normalized = item.file_name.split('#')[0]

            # 우선순위 1: TOC 매핑에서 제목 가져오기
            if item_file_name_normalized in toc_href_to_title and toc_href_to_title[item_file_name_normalized]:
                chapter_title = toc_href_to_title[item_file_name_normalized]
            
            # 우선순위 2: HTML <title> 태그에서 제목 가져오기
            if not chapter_title:
                title_tag = soup.find('title')
                if title_tag and title_tag.string:
                    chapter_title = title_tag.string.strip()
            
            # 우선순위 3: HTML 콘텐츠의 첫 번째 헤딩(h1-h6)에서 제목 가져오기
            if not chapter_title:
                for h_level in range(1, 7): # h1, h2, ..., h6 확인
                    heading_tag = soup.find(f'h{h_level}')
                    if heading_tag and heading_tag.get_text(strip=True):
                        chapter_title = heading_tag.get_text(strip=True)
                        break # 첫 번째 헤딩을 찾으면 중지
            
            # 최종 대체: 제목을 찾지 못하면 "Unnamed Chapter" 할당
            if not chapter_title:
                chapter_title = f"Unnamed Chapter {len(titles)+1}"
            
            # 챕터의 깨끗한 텍스트 콘텐츠 추출
            text = soup.get_text(separator="\n", strip=True)
            
            # 충분한 내용이 있는 챕터만 포함 (예: 100자 이상)
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
    # 업로드된 ePub 파일을 임시 위치에 저장
    with tempfile.NamedTemporaryFile(delete=False, suffix=".epub") as tmp_file:
        tmp_file.write(uploaded_file.read())
        epub_path = tmp_file.name

    with st.spinner("📚 ePub 파일 처리 중..."):
        titles, chapters = extract_epub_chapters(epub_path)

    if not titles:
        st.error("❌ 챕터를 추출할 수 없습니다. ePub 구조를 확인해주세요.")
    else:
        st.success(f"✅ {len(titles)}개의 챕터를 추출했습니다.")
        # 챕터 선택을 위한 드롭다운 메뉴 생성
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
            # 실제 RAG에서는 전체 인덱스를 검색하여 컨텍스트를 찾지만, 여기서는 선택된 챕터의 임베딩만 사용
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
                # 모델 입력 제한을 위해 full_text 길이 조절
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
        # 임시 ePub 파일 정리
        os.remove(epub_path)
    except Exception as e:
        st.error(f"임시 파일 삭제 중 오류 발생: {e}")
