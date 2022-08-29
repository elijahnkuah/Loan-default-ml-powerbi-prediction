from flask import Flask

app = Flask(__name__)

@app.route("/")
def hello():
    return ("This is Elijah !")

@app.route("/try")
def try_work():
    return ("We have to gain it by the Grace of God")

app.run()
