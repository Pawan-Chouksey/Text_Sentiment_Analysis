import os
import pickle
import re
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from BackEnd.TSA_image import extract

try:
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer

    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))
    for word in ["not", "no", "nor", "never", "very", "too"]:
        stop_words.discard(word)
except LookupError:
    lemmatizer = None
    stop_words = set()


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "BackEnd" / "model"
UPLOAD_FOLDER = BASE_DIR / "Uploaded images"
MAX_LEN = 80
LABEL_NAMES = {
    0: "Negative",
    1: "Neutral",
    2: "Positive",
}

app = FastAPI()
app.mount("/static", StaticFiles(directory=BASE_DIR / "FrontEnd" / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "FrontEnd" / "templates")

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

model = load_model(MODEL_DIR / "sentiment_model.keras", compile=False)
with open(MODEL_DIR / "tokenizer.pkl", "rb") as file:
    tokenizer = pickle.load(file)


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = text.split()
    if lemmatizer is not None:
        tokens = [
            lemmatizer.lemmatize(word)
            for word in tokens
            if word not in stop_words
        ]

    return " ".join(tokens)


def prediction_result(user_input: str) -> tuple[str, float, str]:
    cleaned_input = clean_text(user_input)
    if not cleaned_input:
        return "No readable text", 0.0, ""

    sequence = tokenizer.texts_to_sequences([cleaned_input])
    padded_sequence = pad_sequences(sequence, maxlen=MAX_LEN, padding="post", truncating="post",)

    probabilities = model.predict(padded_sequence, verbose=0)[0]
    predicted_label = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_label])
    
    return LABEL_NAMES[predicted_label], confidence, cleaned_input


def render_result(request: Request, source_text: str):
    result, confidence, cleaned_text = prediction_result(source_text)
    return templates.TemplateResponse(
        request,
        "result.html",
        {
            "result": f"{result} ({confidence:.2%})" if confidence else result,
            "input_text": source_text,
            "cleaned_text": cleaned_text,
        },
    )


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {})


@app.post("/predict", response_class=HTMLResponse)
async def predict_text(request: Request, text: str = Form(...)):
    if not text.strip():
        return HTMLResponse(content="No input provided", status_code=400)

    return render_result(request, text)


@app.post("/upload_image", response_class=HTMLResponse)
async def predict_image(request: Request, image: UploadFile = File(...)):
    if not image.filename:
        return HTMLResponse(content="No image uploaded", status_code=400)

    image_path = UPLOAD_FOLDER / Path(image.filename).name
    with open(image_path, "wb") as file:
        file.write(await image.read())

    extracted_text = extract(str(image_path))
    if not extracted_text:
        return HTMLResponse(content="Could not extract text from image", status_code=400)

    return render_result(request, extracted_text)


if __name__ == "__main__":
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8080"))
    print(f"Open http://127.0.0.1:{port} in your browser")
    uvicorn.run(app, host=host, port=port)
