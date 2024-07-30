from flask import Flask,render_template,request
from markupsafe import escape
import pickle
import numpy as np
import jsonify
import jsonify
import joblib
from joblib import dump,load



model_car = joblib.load('filename.pkl') 

app = Flask(__name__)

@app.route('/',methods = ['GET'])
def man():
    return render_template('index.html')

@app.route('/', methods = ['POST'])

def home():
    inp1 = request.form.get("yr")
    inp2 = request.form.get("engine")
    inp3 = request.form.get("cylinder")
    inp4 = request.form.get("city")
    inp5 = request.form.get("highway")
    inp6 = request.form.get("comb")
    inp7 = request.form.get("mg")
    inp8 = request.form.get("emission")
    inp9 = request.form.get("smog")
    inp10 = request.form.get("trans")
    inp11 = request.form.get("type")
    inp12 = request.form.get("vclass")
   
    input_query = np.array([[inp1,inp2,inp3,inp4,inp5,inp6,inp7,inp8,inp9,inp10,inp11,inp12]])
    #print(input_query.dtype)

    #print(inp1,inp2,inp3,inp4,inp5,inp6,inp7,inp8,inp9,inp10,inp11,inp12)
    print(input_query)
    #c = pd.DataFrame(final)
    input_data_as_numpy_array = np.asarray(input_query)

# reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    result = model_car.predict(input_data_reshaped)
    print(result)
    a1 ="Not a good car"
    b1 ="good car"
    return render_template('after.html',data=result)
    
    



if __name__ == '__main__':
    app.run(port=3000,debug=True)