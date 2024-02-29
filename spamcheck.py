import tensorflow_text
import tensorflow as tf
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
from flask import Flask, render_template, request
import numpy as np

model = load_model('Bert_Model', compile=False)
model.load_weights('Weights.hdf5')

app = Flask(__name__)

@app.route('/', methods=["GET", "POST"])
def hello():
    ''' Homepage '''

    return render_template('main.html')

@app.route('/result', methods=["GET", "POST"])
def result():
    ''' Result page '''

    if request.method == "POST":

        opinion = request.form["opinion"]

        x = tf.constant([opinion])
        prediction = np.squeeze(model.predict(x))
        K.clear_session()
        
    return render_template('result.html', prediction=round(float(prediction)*100, 2))

if __name__ == '__main__':
    app.run()