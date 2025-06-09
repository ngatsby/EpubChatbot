def extract_epub_chapters(epub_path):
    book = epub.read_epub(epub_path)
    
    # ITEM_DOCUMENT 대신 MIME 타입으로 직접 필터링
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
                if item.file_name == nav_point.href.split("#")[0]:
                    chapters.append(extract_text_from_item(item))
                    break
        elif isinstance(nav_point, (list, tuple)):
            for sub_point in nav_point:
                parse_nav(sub_point)

    parse_nav(toc)
    return titles, chapters
