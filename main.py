#Import libraries
from flask import Flask, render_template, request
from waitress import serve
from models.Image_Processing import Image_Processing

app = Flask(__name__)
app.config['DEBUG'] == True


#---Define routes---
#Home route
@app.route('/', methods=['GET'])
def home():
    return render_template('home.html')


#Process image route
@app.route('/process_image', methods=['POST'])
def process_image(): 
    Img_Proc = Image_Processing()   
    return Img_Proc.process_image()


serve(app, host='0.0.0.0', port=700)