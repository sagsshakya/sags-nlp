""" 
Name: sags-nlp
File: app.py
-----------------------------
Author: Sagun Shakya
Date:   23/5/2022, 4:32:47 pm
"""

# Imports
from datetime import datetime
from flask import Flask, render_template
from flask_sqlalchemy import SQLAlchemy
from os import makedirs

# Create Flask app.
app = Flask(__name__)

# Home page.
@app.route("/")
def index():
    return render_template("index.html")

# Extractive Summaization.
@app.route("/summarize")
def summarize():
    return render_template("summary.html")

if __name__ == "__main__":
    app.run(debug = True)