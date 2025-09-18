import pytesseract
import numpy as np
from PIL import Image

def extract_text_from_image(image: Image.Image) -> str:
    try:
        img_array = np.array(image)
        text = pytesseract.image_to_string(img_array)
        return text.strip()
    except Exception as e:
        return f"❌ OCR Error: {e}"
