import sys 
from flask import Flask,request ,render_template


from src.exception.exception import CustomeException
from app.prediction import Prediction

app = Flask(__name__)



@app.route("/",methods=["GET","POST"])
def index():
    try:
        if request.method == "GET":
            return render_template("index.html") 
        else:
            message = request.form.get("email")
            if len(message.strip()) == 0:
                return render_template("index.html",result="Please send emial body")
            prediction = Prediction(text=message.strip())
            result = prediction.init_prediction()
            return render_template("index.html",result=result)
    except Exception as e:
        raise CustomeException(e,sys)
        



if __name__ == "__main__":
    app.run(debug=True)
