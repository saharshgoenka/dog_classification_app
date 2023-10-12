from flask import Flask, render_template, request, jsonify
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials
from dotenv import load_dotenv
import os
import io

app = Flask(__name__)

# Load the key and endpoint values from a .env file
load_dotenv()

# Set the values into variables
key = os.getenv('KEY')
endpoint = os.getenv('ENDPOINT')
project_id = os.getenv('PROJECT_ID')
published_name = os.getenv('PUBLISHED_ITERATION_NAME')

# Setup credentials for the client
credentials = ApiKeyCredentials(in_headers={'Prediction-key': key})

# Create the client, which will be used to make predictions
client = CustomVisionPredictionClient(endpoint, credentials)

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        file = request.files['file']

        if file and allowed_file(file.filename):
            # Perform the prediction
            results = classify_image(file)

            return render_template('result.html', results=results)

    return render_template('upload.html')

def classify_image(image_file):
    results = []
    with io.BytesIO(image_file.read()) as image_stream:
        # Perform the prediction
        predictions = client.classify_image(project_id, published_name, image_stream.read())

        for prediction in predictions.predictions:
            results.append({'tag_name': prediction.tag_name, 'probability': prediction.probability})

    return results


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'jpg', 'jpeg', 'png', 'gif', 'bmp'}

if __name__ == '__main__':
    app.run(debug=True)
