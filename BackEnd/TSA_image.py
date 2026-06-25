import cv2 as cv
from easyocr import Reader

reader = None

def get_reader():
    global reader
    if reader is None:
        reader = Reader(['en'])
    return reader

def extract(image_path: str):
    img = cv.imread(image_path)

    if img is None:
        return ""

    grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(grey, (5,5), 0)
    _, thres = cv.threshold(
        blur,
        0,
        255,
        cv.THRESH_BINARY + cv.THRESH_OTSU
    )

    text = get_reader().readtext(thres, detail=0)

    return " ".join(text).strip()