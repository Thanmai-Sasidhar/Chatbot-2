import pytesseract
import numpy as np
import cv2
from PIL import Image

def extract_text_from_image(image: Image.Image) -> str:
    """
    Takes a PIL image and returns extracted text using Tesseract OCR.
    """
    try:
        # Convert PIL image to grayscale OpenCV format
        gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        # Run OCR
        text = pytesseract.image_to_string(gray)
        return text.strip()
    except Exception as e:
        return f"❌ OCR Error: {e}"
