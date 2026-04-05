from flask import Flask, render_template, request, send_file
import os
import pandas as pd
from process import extract_text_from_pdf, match_resume_with_jobs, predict_job_category
from PIL import Image, ImageDraw, ImageFont

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
IMAGE_FOLDER = 'images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['IMAGE_FOLDER'] = IMAGE_FOLDER

# Load job dataset
df = pd.read_csv("nyc-jobs-1.csv")
df = df.dropna(subset=["Additional Information"])

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", process=False, match=None)

@app.route("/process", methods=["GET"])
def process_page():
    return render_template("index.html", process=True, match=None)

@app.route("/", methods=["POST"])
def upload_resume():
    if "resume" not in request.files:
        return "No file part"

    file = request.files["resume"]
    if file.filename == "":
        return "No selected file"

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    resume_text = extract_text_from_pdf(filepath)
    best_match, similarity = match_resume_with_jobs(resume_text, df)
    predicted_category = predict_job_category(resume_text)

    # Generate Image
    image_path = os.path.join(app.config['IMAGE_FOLDER'], f"result_{os.urandom(8).hex()}.png")
    img = Image.new('RGB', (400, 300), color=(50, 50, 50))
    d = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    d.text((20, 20), f"Job: {best_match}", fill=(255, 255, 255), font=font)
    d.text((20, 50), f"Similarity: {similarity}%", fill=(255, 255, 255), font=font)
    img.save(image_path)

    image_url = f"/image/{os.path.basename(image_path)}"

    return render_template("result.html", match=best_match, similarity=similarity, category=predicted_category, image_url=image_url)

@app.route('/image/<filename>')
def serve_image(filename):
    return send_file(os.path.join(app.config['IMAGE_FOLDER'], filename), mimetype='image/png')

if __name__ == "__main__":
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    if not os.path.exists(IMAGE_FOLDER):
        os.makedirs(IMAGE_FOLDER)
    app.run(debug=True, host='0.0.0.0') # Modified line: host='0.0.0.0'