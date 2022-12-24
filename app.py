import numpy as np
from flask import Flask, request, render_template, url_for
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # For rendering results on html UI
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    output = model.predict(final_features)

    result = 'Survived' if output == 1 else 'Failed to Survive'

    return render_template('index.html', prediction_text='This passenger has {}'.format(result))


if __name__ == "__main__":
    app.run(debug=True)
