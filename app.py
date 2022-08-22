from flask import Flask, render_template, request
import ML_Recommendation
app = Flask(__name__,template_folder="templates",static_folder="static")

@app.route("/")
def hello():
    return render_template("index.html")

@app.route("/result", methods = ["POST"])
def submit():
    if request.method =="POST":
        articleID = request.form["articleID"]
        recommendations =[]
        recommendations = ML_Recommendation.cs_recsys.predict([articleID])
    return render_template("result.html",a=articleID, res1 = recommendations[0], res2 = recommendations[1], res3 = recommendations[2], res4 = recommendations[3])

if __name__=="__main__":
    app.run(debug=True)