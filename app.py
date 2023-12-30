# app.py
from logistic_regression import lr_predict_spam
from naive import naive_predict_spam
from flask_cors import CORS
from flask import Flask, render_template, request, flash
from knn import knn_predict_spam

dsNumber=0

classifierApp = Flask(__name__, static_url_path='/static')
CORS(classifierApp)
classifierApp.secret_key = "592123"  # Secret key for Flask application

@classifierApp.route("/")
def index():
    flash("Please enter an email.")
    return render_template("index.html")

@classifierApp.route("/mailController", methods=['POST', 'GET'])
def lr_controller():
    result, test_acc, train_acc, timer_data = lr_predict_spam([str(request.form["mail"])],dsNumber)
    if(result=="-1"):
        flash(f"Please select the database first")
    else:
        flash(f"This email is {result}")
    return render_template("index.html", test_acc=test_acc, train_acc=train_acc, timer_data=timer_data)

@classifierApp.route("/mailController_nb", methods=['POST', 'GET'])
def naive_controller_2():
    result, test_acc, train_acc, timer_data = naive_predict_spam([str(request.form["mail"])],dsNumber)
    if(result=="-1"):
        flash(f"Please select the database first")
    else:
        flash(f"This email is {result}")
    return render_template("index.html", test_acc=test_acc, train_acc=train_acc, timer_data=timer_data)

@classifierApp.route("/mailController_knn", methods=['POST', 'GET'])
def knn_controller():
    result, test_acc, train_acc, timer_data = knn_predict_spam([str(request.form["mail"])],dsNumber)
    flash(f"This email is {result}")
    return render_template("index.html", test_acc=test_acc, train_acc=train_acc, timer_data=timer_data)

@classifierApp.route("/datasetController_1", methods=['POST', 'GET'])
def import_ds_1():
    global dsNumber
    dsNumber = 1
    flash("Using dataset 1")
    return render_template("index.html", test_acc=0, train_acc=0, timer_data=0)

@classifierApp.route("/datasetController_2", methods=['POST', 'GET'])
def import_ds_2():
    global dsNumber
    dsNumber = 2
    flash("Using dataset 2")
    return render_template("index.html", test_acc=0, train_acc=0, timer_data=0)

if __name__ == '__main__':
    classifierApp.run()
