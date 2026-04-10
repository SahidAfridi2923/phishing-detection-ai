from flask import Flask, render_template, request
import os
from predict_url import predict_url

app = Flask(__name__)

# Upload folder (important for deployment)
UPLOAD_FOLDER = "uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    url = request.form["url"]

    image = request.files.get("image")
    image_path = None

    if image and image.filename != "":
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
        image.save(image_path)

    result, confidence, meaning = predict_url(url, image_path)

    return render_template(
        "result.html",
        url=url,
        result=result,
        confidence=confidence,
        meaning=meaning
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)