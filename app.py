from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import pickle

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Load model
with open('model/model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load vectorizer
with open('model/vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        request,
        "index.html"
    )


@app.post("/predict", response_class=HTMLResponse)
async def prediction(request: Request, text: str = Form(...)):

    if text:
        transformed_input = vectorizer.transform([text])
        predicted_sentiment = model.predict(transformed_input)
        result = predicted_sentiment[0]

        return templates.TemplateResponse(
            request,
            "result.html",
            {
                "result": result
            }
        )

    return HTMLResponse(content="No input provided")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
