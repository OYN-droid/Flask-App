import base64
import io
import logging

import pdfplumber
import pytesseract
from openai import OpenAI as OpenAIClient
from PIL import Image

import config

try:
    from langchain_core.documents import Document
except ImportError:  # LangChain < 0.3
    from langchain.schema import Document


logger = logging.getLogger(__name__)

IMAGE_DESCRIPTION_UNAVAILABLE = "[Image description unavailable]"


def extract_pdf_text_for_reading(pdf_path):
    """Extract readable text from a PDF for inline reading."""
    try:
        pages = []
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                text = page.extract_text() or ""
                if text.strip():
                    pages.append(f"Page {page_num}:\n{text}")
        if not pages:
            return "No readable text was found in this PDF."
        return "\n\n".join(pages)
    except Exception:
        logger.exception("Failed to read PDF text from %s", pdf_path)
        return "Failed to read the PDF. Please try again or use a different file."


def describe_image_with_vision(image_bytes):
    """Describe actual image pixels with a vision model, using OCR as a hint."""
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        ocr_text = pytesseract.image_to_string(img).strip()
        image_buffer = io.BytesIO()
        img.save(image_buffer, format="PNG")
        encoded_image = base64.b64encode(image_buffer.getvalue()).decode("ascii")
        prompt = "Describe the visible image concisely and include readable text. Do not invent details."
        if ocr_text:
            prompt += f"\n\nOCR hint (verify against the image):\n{ocr_text}"
        client = OpenAIClient(api_key=config.get_openai_api_key())
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_image}"}},
            ]}],
            max_tokens=500,
        )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("Vision response was empty.")
        return {"subtype": "unknown", "content": content}
    except Exception:
        logger.exception("Failed to describe image")
        return {"subtype": "unknown", "content": IMAGE_DESCRIPTION_UNAVAILABLE}


def extract_text_and_images_from_pdf(pdf_path):
    """Extract text and image descriptions from a PDF."""
    documents = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                text = page.extract_text() or ""
                if text.strip():
                    documents.append(
                        Document(
                            page_content=f"Page {page_num}:\n{text}",
                            metadata={"page": page_num, "type": "text"},
                        )
                    )

                if hasattr(page, "images") and page.images:
                    for img_num, img in enumerate(page.images, 1):
                        try:
                            img_bytes = img.get("stream").get_rawdata()
                            if img_bytes:
                                analysis = describe_image_with_vision(img_bytes)
                                if isinstance(analysis, str):
                                    analysis = {
                                        "subtype": "unknown",
                                        "content": analysis,
                                    }
                                documents.append(
                                    Document(
                                        page_content=(
                                            f"Page {page_num} Image {img_num}:\n"
                                            f"{analysis['content']}"
                                        ),
                                        metadata={
                                            "page": page_num,
                                            "image": img_num,
                                            "type": "image",
                                            "subtype": analysis["subtype"],
                                        },
                                    )
                                )
                        except Exception:
                            logger.exception(
                                "Failed to process image on PDF page %s", page_num
                            )
    except Exception as exc:
        logger.exception("Failed to extract content from PDF %s", pdf_path)
        raise ValueError("Failed to extract content from PDF.") from exc
    return documents
