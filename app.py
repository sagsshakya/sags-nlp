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
from random import randint
from flask import Flask, redirect, render_template, request
from flask_sqlalchemy import SQLAlchemy
from os import makedirs
from os.path import exists, join, isfile
import pandas as pd
from numpy import nan
from warnings import filterwarnings
filterwarnings(action = 'ignore')

# Local Modules.
from utilities.utils import write_id_to_file

# Create a database directory if not present.
makedirs('database', exist_ok = True)

# Load Database.
content_path = join('database', 'content.tsv')

if not isfile(content_path):
    print(f"No file named {content_path}.\nConsider creating a new file in {content_path}...\n")
    # Blank Dataframe.
    dummy_data = {"id" : 0, "date" : "2022-01-01", "text" : "sample text", "summary" : "Sample summary"}
    df_content = pd.DataFrame([list(dummy_data.values())], columns = tuple(dummy_data.keys()))
    df_content.to_csv(content_path, sep = '\t', encoding = "utf-8", index = None)


# Create Flask app.
app = Flask(__name__)

# Home page.
@app.route("/")
@app.route("/index")
def index():
    return render_template("index.html")

# Extractive Summarization.
@app.route("/summarize", methods = ['GET', 'POST'])
def summarize():
    df_content = pd.read_csv(content_path, encoding = 'utf-8', skip_blank_lines = True, delimiter = '\t')

    if request.method == 'POST':

        # Obtain text from the text area.
        text = request.form['input_text']    # Store it in the database if you want.
        assert len(text) > 0, "No text was inserted."

        # Data.

        ## Generate Unique ID for the incoming text.
        flag = True
        while flag:
            text_id = randint(100000, 999999)
            if text_id not in df_content['id']:
                flag = False

        date = datetime.now().date().__str__()
        summary = nan                            # TB modified later.

        # Store info in a dictionary.
        info_dict = {
                        "id" : text_id,
                        "date" : date,
                        "text" : text,
                        "summary" : summary
                    }

        df_content = df_content.append(info_dict, ignore_index = True)

        # Remove dummy row (first row).
        if df_content.iloc[0,0] == 0:
            df_content = df_content.iloc[1:, :]

        # Store in a TSV file.
        df_content.to_csv(content_path, encoding = 'utf-8', header = True, index = None, sep = '\t')

        return redirect("/summarize")

    else:
        current_info = df_content.iloc[-1, :].to_dict()
        return render_template("summary.html", results = current_info)

if __name__ == "__main__":
    app.run(debug = True)