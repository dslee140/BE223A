from flask import Flask, render_template, request, make_response
from flask import Markup
from flask_bootstrap import Bootstrap
from flask_misaka import Misaka
import os
import pandas as pd


content = ""
with open("readme.md", "r") as f:
     content = f.read()


app = Flask(__name__)
bootstrap = Bootstrap(app)
Misaka(app) # To use markdown in the template

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/user/<name>')
def user(name):
    return render_template('user.html',name=name)


@app.route('/home')
def home():
    return render_template('home.html',text=content)

@app.route('/charts')
def chart():
    labels = ["January","February","March","April","May","June","July","August"]
    values = [10,9,8,7,6,4,7,8]
    return render_template('chart.html', values=values, labels=labels)

@app.route('/calendar')
def calendar():
    return render_template("calendar.html")


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)
