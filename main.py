# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.




from flask import Flask,request,jsonify
import pickle
import sklearn

import numpy as np
import jinja2


model = pickle.load(open(r"carmodel.pk1",'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return ("Hello")

@app.route('/predict', methods = ['POST'])

def predict():
    inp1 = request.form.get("inp1")
    inp2 = request.form.get("inp2")
    inp3 = request.form.get("inp3")
    inp4 = request.form.get("inp4")
    inp5 = request.form.get("inp5")
    inp6 = request.form.get("inp6")
    inp7 = request.form.get("inp7")
    inp8 = request.form.get("inp8")
    inp9 = request.form.get("inp9")
    inp10 = request.form.get("inp10")
    inp11 = request.form.get("inp11")
    inp12 = request.form.get("inp12")


    #result = {'inp1':inp1,'inp2':inp2,'inp3':inp3,'inp4':inp4,'inp5':inp5,'inp6':inp6,'inp7':inp7,'inp8':inp8,'inp9':inp9
     #         ,'inp10':inp10,'inp11':inp11,'inp12':inp12}
    input_query = np.array([[inp1,inp2,inp3,inp4,inp5,inp6,inp7,inp8,inp9,inp10,inp11,inp12]])

    result = model.predict(input_query[[0]])

    if result <=5:
        return jsonify("Not A Nature Friedly Car ")
    else:
        return jsonify("Nature Friendly Car")

    return jsonify(str(result))


if __name__ == '__main__':
    app.run(debug=True)
# Press the green button in the gutter to run the script.
#if __name__ == '__main__':
 #   print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
