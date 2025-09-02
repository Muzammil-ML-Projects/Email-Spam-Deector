# libraries
from flask import Flask,request,render_template
import joblib,re,nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

#  Fn. to clean the email text
def textCleaning(text):
    text=text.lower()
    text = re.sub(r"[^a-z\s]","",text)
    text = re.sub(r'https\S+',"",text)
    text = re.sub(r'http\S+',"",text)
    text = re.sub(r'\\S+',"",text)
    text = re.sub(r'\s+'," ",text)
    return text.strip()

# Fn. to tokenize the text
def tokenize(text):
    token = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filter_tokens = [word for word in token if word not in stop_words]
    return filter_tokens

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load("Model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# render index.html
@app.route("/")
def home():
    return render_template("index.html")

# prediction and render to html file
@app.route("/predict", methods=["POST"])
def predict():
    email_text = request.form["email"]

    clean_text = textCleaning(email_text)
    tokens = tokenize(clean_text)
    numeric_token = vectorizer.transform([" ".join(tokens)])
    prediction = model.predict(numeric_token)[0]
    proba = model.predict_proba(numeric_token)[0]
    confidence = round(max(proba) * 100,2) 
    result = "Spam" if prediction == 1 else "Not Spam"
    return render_template("index.html", prediction=result, confidence = confidence)

if __name__ == "__main__":
    app.run(debug=True)


