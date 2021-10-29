# USAGE
# Start the server:
# 	python run_keras_server.py
# Submit a request via cURL:
# 	curl -X POST -F image=@dog.jpg 'http://localhost:5000/predict'
# Submita a request via Python:
#	python simple_request.py

# import the necessary packages
from keras.preprocessing.image import img_to_array
import keras.models
from PIL import Image
import numpy as np
import flask
import io
from flask_cors import CORS, cross_origin

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
model = None


def load_model():
    # load the pre-trained Keras model (here we are using a model
    # pre-trained on ImageNet and provided by Keras, but you can
    # substitute in your own networks just as easily)
    global model
    model = keras.models.load_model("wts_model_v0.h5")
    # model = ResNet50(weights="imagenet")


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
    # image = image.resize(target)
    # image = img_to_array(image)
    # image = np.expand_dims(image, axis=0)
    # image = imagenet_utils.preprocess_input(image)


    # return the processed image
    return image


@app.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False, "result": ""}

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            # read the image in PIL format
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))

            # preprocess the image and prepare it for classification
            image = prepare_image(image, target=(299,299))

            # classify the input image and then initialize the list
            # of predictions to return to the client
            # preds = model.predict(image)
            # data["predictions"] = [preds]
            
            predict = model.predict(image)
            label = ['bull', 'tiger', 'white']
            result = label[np.argmax(predict)]

            data["result"] = result
            data["success"] = True
            print(data)
    # return the data dictionary as a JSON response
    return flask.jsonify(data)


# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
           "please wait until server has fully started"))
    load_model()
    app.run()
