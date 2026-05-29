from flask import Flask, request, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files["file"]
    filename = file.filename
    if filename.endswith('.xlsx') or filename.endswith('.xls'):
        df = pd.read_excel(file)
    else:
        df = pd.read_csv(file)

    vectorizer = TfidfVectorizer()
    x = vectorizer.fit_transform(df["description"])
    y = df["category"]

    model = RandomForestClassifier(n_estimators=100)
    model.fit(x, y)

    df["predicted_category"] = model.predict(x)
    category_totals = df.groupby("predicted_category")["amount"].sum()

    fig, ax = plt.subplots()
    category_totals.plot(kind="pie", autopct="%1.1f%%", ax=ax)
    ax.set_ylabel("")
    ax.set_title("Spending by Category")

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    chart = base64.b64encode(buf.read()).decode("utf-8")
    plt.close()
    return render_template("index.html", chart=chart)

if __name__ == "__main__":
    app.run(debug=True)