# app.py
from logistic_regression import lr_predict_spam
from naive import naive_predict_spam
from flask_cors import CORS
from flask import Flask, render_template, request, flash

classifierApp = Flask(__name__, static_url_path='/static')
CORS(classifierApp)
classifierApp.secret_key = "592123"  # Secret key for Flask application

@classifierApp.route("/")
def index():
    flash("Please enter an email.")
    return render_template("index.html")

@classifierApp.route("/mailController", methods=['POST', 'GET'])
def lr_controller():
    result, test_acc, train_acc = lr_predict_spam([str(request.form["mail"])])
    flash(f"This email is {result}")
    return render_template("index.html", test_acc=test_acc, train_acc=train_acc)

@classifierApp.route("/mailController_2", methods=['POST', 'GET'])
def naive_controller_2():
    result, test_acc, train_acc = naive_predict_spam([str(request.form["mail"])])
    flash(f"This email is {result}")
    return render_template("index.html", test_acc=test_acc, train_acc=train_acc)

@classifierApp.route("/mailController_3", methods=['POST', 'GET'])
def mail_controller_3():
    #result, test_acc, train_acc = naive_predict_spam([str(request.form["mail"])])
    flash(f"This email is a {result} email")
    return render_template("index.html", test_acc=test_acc, train_acc=train_acc)

if __name__ == '__main__':
    classifierApp.run()
