import sys 
from flask import Flask,request ,render_template


from src.exception.exception import CustomeException

app = Flask(__name__)



@app.route("/",methods=["GET"])
def index():
    try:
        return render_template("index.html") 
    except Exception as e:
        raise Exception(e,sys)


@app.route("/prediction",methods=["GET","POST"])
def prediction():
    try:
        if request.method == "GET":
            return render_template("form.html")
        else:
            message = request.form.get("email")
            print(message)
            return render_template("form.html",result=message)
    except Exception as e:
        raise CustomeException(e,sys)



if __name__ == "__main__":
    app.run(debug=True)
