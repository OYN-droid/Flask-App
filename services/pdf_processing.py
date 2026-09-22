import base64
import io
import json
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


def normalize_image_subtype(classification):
    """Normalize the vision model's classification to stable metadata values."""
    classification = str(classification or "").strip().lower()
    if "chart" in classification or "graph" in classification:
        return "chart"
    if "table" in classification:
        return "table"
    if "diagram" in classification:
        return "diagram"
    if "photo" in classification or "logo" in classification:
        return "photo"
    return "unknown"


def parse_vision_analysis(response_content):
    """Parse the model's JSON response into indexed content and image subtype."""
    response_content = (response_content or "").strip()
    if response_content.startswith("```"):
        response_content = response_content.removeprefix("```json").removeprefix("```")
        response_content = response_content.removesuffix("```").strip()

    analysis = json.loads(response_content)
    subtype = normalize_image_subtype(analysis.get("classification"))
    content = analysis.get("content", "")
    if not isinstance(content, str):
        content = json.dumps(content, ensure_ascii=False, indent=2)
    if not content.strip():
        raise ValueError("Vision response did not contain image content.")
    return {"subtype": subtype, "content": content.strip()}


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
    """Classify and analyze an image, using local OCR as a supplementary hint."""
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        ocr_text = pytesseract.image_to_string(img).strip()

        image_buffer = io.BytesIO()
        img.save(image_buffer, format="PNG")
        encoded_image = base64.b64encode(image_buffer.getvalue()).decode("ascii")

        prompt = """Analyze this image and return a JSON object with exactly two fields:
"classification" and "content".

Classify the image as exactly one of: "photo/logo", "chart or graph", "table", or
"diagram". Format "content" according to that classification:
- For "photo/logo", give the same kind of direct, concise prose description you
  would normally provide, including important objects, logos, and readable text.
- For "table", transcribe the underlying data as a markdown table. Preserve headers,
  row labels, values, units, and footnotes as accurately as the image allows.
- For "chart or graph", provide a structured markdown list containing the chart type,
  title, axis labels and units, legend/series, approximate data points, and overall trend.
- For "diagram", concisely describe its labeled components, relationships, and flow.

Use the actual image as the source of truth. Do not invent unreadable values."""
        if ocr_text:
            prompt += (
                "\n\nLocal OCR extracted the following supplementary text. Verify it "
                f"against the image rather than relying on it alone:\n\n{ocr_text}"
            )
        else:
            prompt += "\n\nLocal OCR did not extract any readable text."

        client = OpenAIClient(api_key=config.get_openai_api_key())
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{encoded_image}"
                            },
                        },
                    ],
                }
            ],
            max_tokens=1000,
            response_format={"type": "json_object"},
        )
        try:
            return parse_vision_analysis(response.choices[0].message.content)
        except Exception:
            logger.exception("Vision response did not contain valid classified content")
            return {
                "subtype": "unknown",
                "content": IMAGE_DESCRIPTION_UNAVAILABLE,
            }
    except Exception:
        logger.exception("Failed to describe image")
        return {
            "subtype": "unknown",
            "content": IMAGE_DESCRIPTION_UNAVAILABLE,
        }


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
