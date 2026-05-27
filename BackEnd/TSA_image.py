import cv2 as cv
from easyocr import Reader

reader = Reader(['en'])

def extract(image_path: str):

    img = cv.imread(image_path)

    if img is None:
        return ""

    grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(grey, (5,5), 0)
    _, thres = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    text = reader.readtext(thres, detail=0)

    return " ".join(text).strip()