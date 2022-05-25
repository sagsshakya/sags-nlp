""" 
Name: sags-nlp
File: app.py
-----------------------------
Author: Sagun Shakya
Date:   23/5/2022, 4:32:47 pm
"""

# Imports
from datetime import datetime
from genericpath import isfile
import os
from random import randint
from flask import Flask, redirect, render_template, request
from flask_sqlalchemy import SQLAlchemy
from os import makedirs
from os.path import exists, join, isfile
import pandas as pd

from utilities.utils import write_id_to_file, write_to_file

# Create Flask app.
app = Flask(__name__)

# Home page.
@app.route("/")
@app.route("/index")
def index():
    return render_template("index.html")

# Extractive Summaization.
@app.route("/summarize", methods = ['GET', 'POST'])
def summarize():

    if request.method == 'POST':

        # Obtain text from the text area.
        text = request.form['input_text']    # Store it in the database if you want.

        # Create a database directory if not present.
        makedirs('database', exist_ok = True)
        id_path = join('database', 'IDs.txt')

        # Generate a unique ID for an input text and store it in a text file.
        text_id = write_id_to_file(filepath = id_path)

        return redirect("/summarize")

    else:     
        return render_template("summary.html")

if __name__ == "__main__":
    app.run(debug = True)