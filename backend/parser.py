import os

from dotenv import load_dotenv
from pypdf import PdfReader

load_dotenv()
# Explicit opt-in: a LLAMA_CLOUD_API_KEY in .env alone must not send resumes to a cloud service.
PDF_PARSERS = ("pypdf", "llamaparse")
PDF_PARSER = os.getenv("PDF_PARSER", "pypdf")
if PDF_PARSER not in PDF_PARSERS:
    raise ValueError(f"PDF_PARSER must be one of {PDF_PARSERS}, got {PDF_PARSER!r}")
MAX_PAGES = 10  # a resume never needs more; bounds the work a crafted PDF can cause


def parse_resume(file_path: str) -> str | None:
    if PDF_PARSER == "llamaparse":
        try:
            text = _parse_with_llamaparse(file_path)
            if text.strip():
                return text
        except Exception as e:
            # quota, network or key problems shouldn't make a readable PDF look broken
            print(f"LlamaParse failed, falling back to pypdf: {e}")

    try:
        return "\n".join(page.extract_text() or "" for page in PdfReader(file_path).pages[:MAX_PAGES])
    except Exception as e:
        print(f"Error parsing the pdf: {e}")
        return None


def _parse_with_llamaparse(file_path: str) -> str:
    # imported lazily so the default local path doesn't need the cloud SDK
    import nest_asyncio
    from llama_parse import LlamaParse

    nest_asyncio.apply()
    documents = LlamaParse(result_type="markdown").load_data(file_path)
    return "\n".join(doc.text for doc in documents)
