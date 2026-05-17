from pathlib import Path
import fitz

from langchain_community.document_loaders import PyMuPDFLoader


PDF_PATH = "data/documents/wiser-provider-supplier-guide.pdf"


def extract_pdf_metadata(pdf_path: str):
    """
    Extract detailed PDF metadata using PyMuPDF.
    """

    pdf_document = fitz.open(pdf_path)

    metadata = pdf_document.metadata

    total_pages = pdf_document.page_count

    has_images = False

    for page_num in range(total_pages):
        page = pdf_document.load_page(page_num)

        images = page.get_images()

        if images:
            has_images = True
            break

    print("\n========== PDF METADATA ==========")

    print(f"📄 File Name       : {Path(pdf_path).name}")
    print(f"📚 Total Pages     : {total_pages}")
    print(f"🖼️ Contains Images : {has_images}")
    print(f"🔒 Is Encrypted    : {pdf_document.is_encrypted}")

    print("\n📌 Document Metadata:")

    for key, value in metadata.items():
        print(f"{key}: {value}")

    print("\n==================================\n")

    pdf_document.close()


def load_pdf(pdf_path: str):
    """
    Load PDF using LangChain PyMuPDFLoader.
    """

    path = Path(pdf_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    extract_pdf_metadata(pdf_path)

    print(f"📥 Loading PDF: {path.name}")

    loader = PyMuPDFLoader(str(path))

    documents = loader.load()

    print("\n✅ PDF loaded successfully")
    print(f"📄 Total LangChain Documents: {len(documents)}")

    print("\n📌 First Page Metadata:")
    print(documents[0].metadata)

    print("\n📌 First 500 Characters:\n")
    print(documents[0].page_content[:500])

    return documents


if __name__ == "__main__":
    load_pdf(PDF_PATH)