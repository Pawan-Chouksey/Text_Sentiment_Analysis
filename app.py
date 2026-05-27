from flask import Flask, request, render_template
import pickle
import os
from BackEnd.TSA_image import extract

TSA_Model= Flask(__name__)

UPLOAD_FOLDER = "Uploaded images"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
TSA_Model.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

with open('BackEnd/model/model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('BackEnd/model/vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)


def prediction_result(user_input: str):
    transformed_input = vectorizer.transform([user_input])
    predicted_sentiment = model.predict(transformed_input)
    result = predicted_sentiment[0]
    return result

@TSA_Model.route('/', methods=['GET', 'POST'])
def index():
    return render_template('FrontEnd/templates/index.html')


@TSA_Model.route('/predict', methods=['POST'])
def prediction():
    if request.method == 'POST':

        user_input = request.form.get('text')

        if user_input: 
            result = prediction_result(user_input)
            return render_template('FrontEnd/templates/result.html', result=result)
        else:
            return "No input provided"    


@TSA_Model.route('/upload_image', methods=['POST'])
def upload_img():

    file = request.files.get('image')

    if file:
        image_path = os.path.join(TSA_Model.config['UPLOAD_FOLDER'], file.filename)
        file.save(image_path)

        text_extracted = extract(image_path)

        if text_extracted:
            result = prediction_result(text_extracted)
            return render_template('FrontEnd/templates/result.html', result=result, input_text=text_extracted)
        else:
            return "Could not extract text from image"
    else:
        return "No file Uploaded"


if __name__ == '__main__':
    TSA_Model.run(host='0.0.0.0', port=5000)
