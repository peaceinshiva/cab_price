import pickle
from flask import Flask, render_template, request


import os

model_reloaded = pickle.load(open('static/Our_Trained_cab_model.sav', 'rb'))

name_reloaded = pickle.load(open('static/Our_name_encode.sav', 'rb'))
so_reloaded = pickle.load(open('static/Our_so_encode.sav', 'rb'))
desti_reloaded = pickle.load(open('static/Our_desti_encode.sav', 'rb'))

app=Flask(__name__)
UPLOAD_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')

def upload():
    return render_template("cab.html")

            
  
@app.route('/fildetails', methods=['GET','POST'])
def fildetails():
    if request.method=='POST':
        name=request.form['name']
        distance=request.form['distance']
        surge_mu=request.form['multiplier']
        source=request.form['source']
        destination=request.form['destination']
        # data=name+distance+source+destination+surge_mu

        p=model_reloaded.predict([[distance,name_reloaded.transform([name]),surge_mu,so_reloaded.transform([source]),desti_reloaded.transform([destination])]])
        return render_template("cab.html",prd=float(p))
                
            
if __name__ == '__main__':
    app.run()

