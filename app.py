from flask import Flask, render_template, request, redirect, url_for, jsonify, send_file
from predict import predict_disease
from werkzeug.utils import secure_filename
import os
import sqlite3
import csv
from datetime import datetime

# ==========================
# Config Flask
# ==========================
app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

DB_FILE = "history.db"

# ==========================
# Initialize Database
# ==========================
def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            image TEXT,
            result TEXT,
            confidence REAL
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# ==========================
# Helper functions
# ==========================
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def get_history():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM history ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()
    history_data = []
    counts = {"Blast":0,"Blight":0,"Brown Spot":0,"Hispa":0,"Normal":0,"Tungro":0}
    for row in rows:
        item = {
            "id": row[0],
            "date": row[1],
            "thumbnail": row[2],
            "result": row[3],
            "confidence": row[4]
        }
        history_data.append(item)
        if row[3] in counts:
            counts[row[3]] += 1
    chart_data = [
        counts["Blast"],
        counts["Blight"],
        counts["Brown Spot"],
        counts["Hispa"],
        counts["Normal"],
        counts["Tungro"]
    ]
    return history_data, chart_data

# ==========================
# Route: Disease Detection
# ==========================
@app.route("/", methods=["GET","POST"])
def detect():
    result = None
    confidence = None
    img_path = None
    error = None

    if request.method == "POST":
        if "image" not in request.files:
            error = "File tidak ditemukan."
        else:
            file = request.files["image"]
            if file.filename == "":
                error = "Silakan pilih gambar."
            elif not allowed_file(file.filename):
                error = "Format harus JPG/JPEG/PNG."
            else:
                filename = secure_filename(file.filename)
                img_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                file.save(img_path)

                try:
                    result, confidence = predict_disease(img_path)

                    # Simpan ke database
                    conn = sqlite3.connect(DB_FILE)
                    cursor = conn.cursor()
                    cursor.execute("INSERT INTO history (date,image,result,confidence) VALUES (?,?,?,?)",
                                   (datetime.now().strftime("%d-%m-%Y %H:%M"), img_path, result, round(confidence,2)))
                    conn.commit()
                    conn.close()

                except Exception as e:
                    error = str(e)

    return render_template("index.html",
                           result=result,
                           confidence=confidence,
                           img_path=img_path,
                           error=error)

# ==========================
# Route: Detection History
# ==========================
@app.route("/history")
def history():
    history_data, chart_data = get_history()
    return render_template("history.html", history=history_data, chart_data=chart_data)

# ==========================
# Route: Delete Record
# ==========================
@app.route("/delete/<int:record_id>", methods=["POST"])
def delete_history(record_id):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM history WHERE id=?", (record_id,))
    conn.commit()
    conn.close()
    return jsonify({"success": True})

# ==========================
# Route: Export CSV
# ==========================
@app.route("/export_csv")
def export_csv():
    history_data, _ = get_history()
    csv_file = "history_export.csv"
    with open(csv_file,"w",newline="",encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ID","Tanggal","Image","Penyakit","Confidence"])
        for row in history_data:
            writer.writerow([row["id"],row["date"],row["thumbnail"],row["result"],row["confidence"]])
    return send_file(csv_file, as_attachment=True)

# ==========================
# Route: Disease Information
# ==========================
@app.route("/information")
def information():
    diseases = [
        {"name":"Blast","image":"/static/images/blast.jpg","description":"Penyakit akibat jamur Magnaporthe oryzae.","symptoms":"Bercak berbentuk belah ketupat pada daun.","causes":"Infeksi jamur Magnaporthe oryzae.","treatment":"Gunakan fungisida dan varietas tahan penyakit."},
        {"name":"Blight","image":"/static/images/blight.jpg","description":"Hawar daun bakteri pada tanaman padi.","symptoms":"Daun menguning lalu mengering.","causes":"Bakteri Xanthomonas oryzae.","treatment":"Gunakan benih sehat dan sanitasi lahan."},
        {"name":"Brown Spot","image":"/static/images/brown_spot.jpg","description":"Penyakit bercak coklat pada daun.","symptoms":"Muncul bercak coklat bulat pada daun.","causes":"Jamur Bipolaris oryzae.","treatment":"Pemupukan seimbang dan fungisida."},
        {"name":"Hispa","image":"/static/images/hispa.jpg","description":"Kerusakan akibat serangan hama hispa.","symptoms":"Daun terlihat bergaris putih.","causes":"Serangan serangga hispa.","treatment":"Gunakan insektisida sesuai dosis."},
        {"name":"Normal","image":"/static/images/normal.jpg","description":"Daun padi sehat.","symptoms":"Tidak ada gejala penyakit.","causes":"-","treatment":"Pertahankan perawatan tanaman."},
        {"name":"Tungro","image":"/static/images/tungro.jpg","description":"Penyakit virus pada padi.","symptoms":"Daun menguning dan kerdil.","causes":"Virus Tungro dibawa wereng hijau.","treatment":"Pengendalian wereng dan varietas tahan."}
    ]
    return render_template("info.html", diseases=diseases)

# ==========================
# MAIN
# ==========================
if __name__ == "__main__":
    app.run(debug=True)