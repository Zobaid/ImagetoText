from flask import Flask, request
import os

class Image_Processing:
    def process_image(self):
        #return request.form['txt-filename']
        app = Flask(__name__)
        app.config['Image_Uploads'] = 'static/images'
        if request.files:
            image = request.files['img']
            #return image.filename
            image.save(os.path.join( app.config['Image_Uploads'] , image.filename))
            return 'file ' + image.filename + ' has been saved'