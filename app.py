# USAGE
# Start the local server:
# 	python app.py
# Submit a request via cURL:
# 	curl -X POST -F image=@testshark.jpg 'https://whattheshark-backend.herokuapp.com/predict'

# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from PIL import Image
import numpy as np
from flask import Flask,request,jsonify
import io
import gc
import keras.backend as K
from flask_cors import CORS, cross_origin

# initialize our Flask application and the Keras model
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
model = load_model("wts_model_v1.h5")



def prepare_image(image, target):
    # if the image mode is not RGB, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # resize the input image and preprocess it
    image = image.resize(target)
    image = np.array(image)
    image = image.astype('float32')
    image /= 255
    image = np.reshape(image ,(1,299,299,3))


    # return the reshaped image
    return image

@app.route("/",methods=["GET"])
def main():
    return "What The Shark Backend/Deep Learning"
    
@app.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False, "result": ""}

    # ensure an image was properly uploaded to our endpoint
    if request.method == "POST":
        if request.files.get("image"):
            # read the image in PIL format
            
            image = request.files["image"].read()
            image = Image.open(io.BytesIO(image))

            # preprocess the image and prepare it for classification
            image = prepare_image(image, target=(299,299))

            
            # prediction = model.predict(image)
            # predict_on_batch() for prevent memory leak
            prediction = model.predict(image)
            label = ['bull', 'tiger', 'white']
            result = label[np.argmax(prediction)]

            data["result"] = result
            data["success"] = True
            print(data)
            gc.collect()
            K.clear_session()
    # return the data dictionary as a JSON response
    return jsonify(data)


# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
           "please wait until server has fully started"))
    app.run()
