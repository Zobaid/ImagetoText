import flask
from flask import request, redirect,render_template
from werkzeug.utils import secure_filename

import os

app = flask.Flask(__name__)

app.config["IMAGE_UPLOADS"] = "G:/Projects/SimpleHTR/SimpleHTR/input_words"

@app.route("/upload-image", methods=["GET", "POST"])
def upload_image():

    if request.method == "POST":

        if request.files:

            image = request.files["image"]

            image.save(os.path.join(app.config["IMAGE_UPLOADS"], image.filename))

            print("Image saved")

            return redirect(request.url)

    return render_template("/templates/upload_image.html")
app.run(debug=True)