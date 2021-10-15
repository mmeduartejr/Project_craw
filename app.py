import flask
import pickle
import pandas as pd
# Load NeuroKit and other useful packages
import neurokit2 as nk
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from time import time
from keras.models import load_model

IMAGE_FOLDER = './static/waveforms/'


model = load_model(r"./model/ECG Classifier.h5")
app = flask.Flask(__name__, template_folder='pages')
app.config['UPLOAD_FOLDER'] = IMAGE_FOLDER
plt.rcParams['figure.figsize'] = [10, 6]  # Bigger images



@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return(flask.render_template('main.html'))   
    if flask.request.method == 'POST':
        noise = flask.request.form['noise']
        heart_rate = flask.request.form['heart_rate']
        ecg_noise= float(noise)
        ecg_heart_rate= int(heart_rate)
        print("noise is:" + str(ecg_noise))
        print("heart rate is:" + str(ecg_heart_rate))
        ecg = nk.ecg_simulate(duration=10, noise=ecg_noise, heart_rate=ecg_heart_rate)
        ecg_df = pd.DataFrame({"ecg": ecg})
        plt.plot(ecg_df)
        new_graph_name = "plot" + str(time()) + ".png"
        for filename in os.listdir('static/waveforms/'):
            if filename.startswith('plot'):  # not to remove other images
                os.remove('static/waveforms/' + filename)
        plt.savefig('./static/waveforms/'+new_graph_name)
        plt.show()
        full_filename = os.path.join(app.config['UPLOAD_FOLDER'], new_graph_name)
        ecg_input = ecg[0:187]
        ecg_input.shape[0]
        ecg_input = np.array(ecg_input).reshape(1, ecg_input.shape[0], 1)
        true_pred = model.predict(ecg_input)
        array= true_pred[0].tolist()
        max_value = max(array)
        max_index = array.index(max_value)
        print(max_value)
        print(max_index)
        print(true_pred[0])
        heart_list=['NORMAL','SVEB','VEB','FUSION BEAT','UNKNOWN BEAT']
        result=heart_list[max_index]
        print(result)
        return flask.render_template("main.html",original_input={'noise': noise,
                                                     'heart_rate': heart_rate}, user_image = full_filename, result=str(result))


if __name__ == '__main__':
    app.run() 