from flask import Flask, render_template, jsonify, request, send_file
from src.exception import CustomException
from src.logger import logging as lg
import os,sys

from src.pipelines.train_pipeline import TrainPipeline
from src.pipelines.prediction_pipeline import PredictionPipeline

app = Flask(__name__)

@app.route("/")
def home():
    return jsonify("Welcome To Home Page")


@app.route("/train")
def train_route():
    try:
        train_pipeline = TrainPipeline()
        train_pipeline.run_pipeline()

        return "Training Completed."

    except Exception as e:
        raise CustomException(e,sys)

@app.route('/upload', methods=['POST', 'GET'])
def upload():
    
    try:


        if request.method == 'POST':
            prediction_pipeline = PredictionPipeline(request)
            prediction_file_config = prediction_pipeline.run_pipeline()

            lg.info("prediction completed. Downloading prediction file.")
            return send_file(prediction_file_config.prediction_file_path,
                            download_name= prediction_file_config.prediction_file_name,
                            as_attachment= True)


        else:
            return render_template('index.html')
    except Exception as e:
        raise CustomException(e,sys)
    


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug= True)