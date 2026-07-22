from pypdf import PdfReader


def extract_text(pdf_path: str) -> str:
    """샘플 제안서 PDF에서 텍스트를 추출한다. 세 AI 모두 동일한 입력을 받도록
    (프로바이더별 문서 처리 방식 차이를 없애기 위해) 텍스트로 통일한다."""
    reader = PdfReader(pdf_path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages).strip()
