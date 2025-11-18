from flask import Flask, render_template, request, redirect, session
import sqlite3
import random
import requests
from datetime import datetime, timedelta
import calendar

app = Flask(__name__)
app.secret_key = "secret123"

def get_db():
    conn = sqlite3.connect("database.db")
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    cursor = conn.cursor()

    # tabel users
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT,
            nickname TEXT UNIQUE
        )
    """)

    # tabel pertanyaan
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS questions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT,
            option_a TEXT,
            option_b TEXT,
            option_c TEXT,
            option_d TEXT,
            correct TEXT
        )
    """)

    # tabel leaderboard
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS leaderboard (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            nickname TEXT,
            score INTEGER
        )
    """)

    # cek apakah sudah ada data
    # cursor.execute("DELETE FROM questions")
    cursor.execute("SELECT COUNT(*) FROM questions")
    count = cursor.fetchone()[0]

    if count == 0:
        sample_questions = [
            # --- AI Development in Python ---
            ("Library Python yang paling umum digunakan untuk machine learning adalah?", "NumPy", "Pandas", "Scikit-learn", "Matplotlib", "C"),
            ("Apa fungsi utama TensorFlow?", "Membuat website", "Pemrosesan teks", "Machine learning & deep learning", "Membuat game", "C"),
            ("Framework deep learning yang dikembangkan oleh Facebook adalah?", "TensorFlow", "Caffe", "PyTorch", "Keras", "C"),
            ("Model yang memprediksi nilai kontinu disebut?", "Classification", "Clustering", "Regression", "Segmentation", "C"),
            ("Algoritma yang digunakan untuk clustering adalah?", "Linear Regression", "KNN", "K-Means", "Naive Bayes", "C"),
            ("Proses membagi data menjadi train dan test disebut?", "Splitting", "Separation", "Sampling", "Normalization", "A"),
            ("Library yang digunakan untuk manipulasi array numerik adalah?", "Matplotlib", "NumPy", "Seaborn", "Flask", "B"),
            ("Metode untuk meningkatkan akurasi model dengan banyak model disebut?", "Stacking", "Boosting", "Bagging", "Pruning", "C"),
            ("Hyperparameter biasanya di-tuning menggunakan?", "GridSearchCV", "train_test_split", "fit()", "compile()", "A"),
            ("Fungsi aktivasi umum pada hidden layer adalah?", "ReLU", "Softmax", "Sigmoid", "Linear", "A"),

            # --- Computer Vision ---
            ("Library Python untuk visi komputer adalah?", "OpenCV", "Flask", "Requests", "SQLAlchemy", "A"),
            ("Konvolusi dalam CNN digunakan untuk?", "Memperbesar gambar", "Mengambil fitur visual", "Menghapus noise", "Mengganti warna", "B"),
            ("Pooling layer berfungsi untuk?", "Menambah resolusi", "Menurunkan dimensi fitur", "Menambah piksel", "Menambah warna", "B"),
            ("Cascade classifier digunakan untuk mendeteksi?", "Objek dalam teks", "Wajah", "Audio", "Suara manusia", "B"),
            ("Format warna default OpenCV adalah?", "RGB", "CMYK", "HSV", "BGR", "D"),
            ("YOLO digunakan untuk?", "Klasifikasi teks", "Deteksi objek real-time", "Analisis suara", "Prediksi cuaca", "B"),
            ("Image augmentation berguna untuk?", "Mengurangi dataset", "Menambah variasi data", "Menghapus noise", "Menyimpan model", "B"),
            ("Pre-trained model untuk deteksi wajah?", "VGG16", "Haar Cascade", "Inception", "LeNet", "B"),
            ("Operasi blur pada gambar menggunakan?", "Kernel", "Layer", "Dataset", "Optimizer", "A"),
            ("Model CNN pertama yang populer adalah?", "AlexNet", "ResNet", "MobileNet", "VGG19", "A"),

            # --- NLP (Natural Language Processing) ---
            ("Library NLP populer di Python adalah?", "BeautifulSoup", "spaCy", "Django", "OpenCV", "B"),
            ("Tokenization berarti?", "Menghapus kata", "Memecah teks menjadi unit kecil", "Menggabungkan kalimat", "Membersihkan data", "B"),
            ("Stopword adalah?", "Kata yang penting", "Kata yang tidak berarti", "Jenis kata kerja", "Model bahasa", "B"),
            ("Model BERT digunakan untuk?", "Membuat game", "Pengolahan bahasa alami", "Analisis suara", "Deteksi objek", "B"),
            ("Stemming bertujuan untuk?", "Mengubah kata ke bentuk dasar", "Menghapus huruf vokal", "Membalik kalimat", "Menambah kata baru", "A"),
            ("Sentiment analysis digunakan untuk?", "Mendeteksi objek", "Menilai emosi pada teks", "Mengklasifikasi gambar", "Mengubah audio", "B"),
            ("Bag-of-Words adalah?", "Model statistik teks", "Model gambar", "Model suara", "Model numerik", "A"),
            ("Word Embedding menghasilkan?", "Angka acak", "Representasi kata berbentuk vektor", "Gambar", "Audio", "B"),
            ("Model GPT termasuk jenis?", "CNN", "RNN", "Transformer", "GAN", "C"),
            ("TF-IDF menghitung?", "Frekuensi relatif kata", "Jumlah dataset", "Jumlah layer", "Jumlah neuron", "A"),

            # --- Deploying AI to Applications ---
            ("Framework yang umum dipakai untuk membuat API model AI?", "Flask", "Excel", "Photoshop", "Unity", "A"),
            ("File model TensorFlow biasanya berekstensi?", ".csv", ".h5", ".exe", ".sql", "B"),
            ("Flask route digunakan untuk?", "Menyimpan database", "Menampilkan halaman web", "Mengubah dataset", "Menyimpan gambar", "B"),
            ("Model inference berarti?", "Training model", "Menjalankan prediksi", "Membersihkan data", "Menghapus file", "B"),
            ("Library untuk membuat restful API di Python?", "Pandas", "Flask", "NumPy", "Matplotlib", "B"),
            ("Model AI sering di-deploy menggunakan format?", "SavedModel", "JPEG", "WAV", "PNG", "A"),
            ("Apa fungsi joblib?", "Menampilkan grafik", "Menyimpan & load model", "Mengatur server", "Membuat gambar", "B"),
            ("Streamlit digunakan untuk?", "Membuat antarmuka AI", "Melatih CNN", "Mendeteksi objek", "Membuat audio", "A"),
            ("TensorFlow Lite berfungsi untuk?", "Mobile deployment", "Web scraping", "Image editing", "Database query", "A"),
            ("Flask menerima input dari user melalui?", "Route", "Dataset", "Request", "Optimizer", "C"),

            # --- More AI/Python ---
            ("Epoch dalam training adalah?", "Satu kali lewat seluruh data", "Satu batch", "Satu neuron", "Satu folder", "A"),
            ("Overfitting terjadi ketika?", "Model terlalu sederhana", "Model terlalu menghafal data", "Dataset terlalu besar", "Model tidak dilatih", "B"),
            ("Regularization bertujuan untuk?", "Membuat model besar", "Mengurangi overfitting", "Membuat dataset", "Menghapus model", "B"),
            ("Dropout digunakan untuk?", "Menambah layer", "Menghapus sebagian neuron saat training", "Menambah dataset", "Menghapus folder", "B"),
            ("Optimizer yang populer untuk deep learning?", "Adam", "RMSProp", "SGD", "Semua benar", "D"),
            ("Batch size adalah?", "Jumlah neuron", "Jumlah input dalam sekali forward", "Jumlah layer", "Jumlah label", "B"),
            ("Confusion matrix digunakan untuk?", "Mengukur performa klasifikasi", "Membuat grafik", "Menambah data", "Menghapus error", "A"),
            ("Precision mengukur?", "Prediksi yang benar dari prediksi positif", "Jumlah data", "Kecepatan model", "Epoch", "A"),
            ("Recall mengukur?", "Akurasi keseluruhan", "Prediksi benar dari data aktual positif", "Kecepatan GPU", "Waktu training", "B"),
            ("Loss function digunakan untuk?", "Mengukur error model", "Menambah data", "Mengedarkan dataset", "Membuat grafik", "A"),

            # --- Computer Vision advanced ---
            ("Model CNN yang memenangkan ILSVRC 2015 adalah?", "AlexNet", "VGG", "ResNet", "LeNet", "C"),
            ("GAN digunakan untuk?", "Mendeteksi objek", "Menghasilkan data baru", "Menambah loss", "Memotong gambar", "B"),
            ("Edge detection dilakukan dengan?", "Sobel", "Pooling", "Dropout", "Embedding", "A"),
            ("Preprocessing gambar biasanya melibatkan?", "Resize", "Rotate", "Normalize", "Semua benar", "D"),
            ("Segmentation bertujuan untuk?", "Memisahkan objek dalam gambar", "Menyimpan gambar", "Menghapus resolusi", "Memutar foto", "A"),
            ("Heatmap digunakan untuk?", "Visualisasi lokasi fitur penting", "Menghapus noise", "Membersihkan dataset", "Menambah warna", "A"),
            ("MobileNet dirancang untuk?", "Server mahal", "Perangkat mobile", "Superkomputer", "Game engine", "B"),
            ("OpenCV function cv2.imread digunakan untuk?", "Membaca gambar", "Menulis gambar", "Memotong gambar", "Resize", "A"),
            ("Grayscale berarti gambar memiliki?", "3 channel", "1 channel", "4 channel", "2 channel", "B"),
            ("FPS berarti?", "Frame per second", "Foto per second", "Filter per second", "Function per second", "A"),

            # --- NLP advanced ---
            ("LSTM adalah jenis?", "CNN", "RNN", "Transformer", "GAN", "B"),
            ("Attention digunakan dalam?", "Transformer", "GAN", "CNN", "VAE", "A"),
            ("NER berarti?", "Natural Error Rate", "Named Entity Recognition", "New Entity Rule", "Neutral Emotion Reading", "B"),
            ("HuggingFace menyediakan?", "Model NLP", "Dataset", "Tokenizer", "Semua benar", "D"),
            ("Embedding model Word2Vec dikembangkan oleh?", "Google", "OpenAI", "Facebook", "Amazon", "A"),
            ("Seq2Seq digunakan untuk?", "Machine translation", "Image editing", "Object detection", "Audio cleaning", "A"),
            ("BLEU score dipakai untuk?", "Evaluasi terjemahan teks", "Evaluasi gambar", "Evaluasi suara", "Evaluasi video", "A"),
            ("Transformers diperkenalkan oleh paper?", "DeepFace", "Attention Is All You Need", "AlphaZero", "ImageNet", "B"),
            ("Stopword biasanya dihapus karena?", "Tidak penting secara makna", "Sulit dibaca", "Mengandung angka", "Tidak dapat diproses", "A"),
            ("Sentiment polarity terdiri dari?", "Positif, negatif, netral", "Baik, buruk", "Hitam, putih", "Tinggi, rendah", "A"),

            # --- AI deployment advanced ---
            ("REST API berbasis JSON digunakan untuk?", "Mengembalikan data ke user", "Menyimpan gambar", "Membuat GPU", "Membersihkan dataset", "A"),
            ("Flask digunakan terutama untuk?", "Backend web", "Frontend web", "Machine learning", "Editing video", "A"),
            ("Uvicorn dan Gunicorn digunakan untuk?", "Menjalankan API", "Membuat model", "Menggambar grafik", "Scraping", "A"),
            ("FastAPI lebih cepat dari Flask karena?", "Menggunakan ASGI", "Menghapus HTML", "Tidak pakai Python", "Tanpa server", "A"),
            ("Streamlit dapat menampilkan?", "Chart", "Input form", "Gambar", "Semua benar", "D"),
            ("Docker digunakan untuk?", "Containerizing aplikasi", "Menambah dataset", "Editing HTML", "Membuat grafik", "A"),
            ("File `.pkl` biasanya untuk?", "Model Python", "Gambar", "HTML", "Audio", "A"),
            ("Model inference biasanya butuh?", "Model + input data", "GPU kuat", "Data training", "Excel", "A"),
            ("Deployment AI membutuhkan?", "Server", "Model", "API", "Semua benar", "D"),
            ("Model AI yang sangat besar sering disebut?", "Tiny model", "Nano model", "Large Language Model", "Mini model", "C")
        ]

        cursor.executemany("""
            INSERT INTO questions (question, option_a, option_b, option_c, option_d, correct)
            VALUES (?, ?, ?, ?, ?, ?)
        """, sample_questions)

    conn.commit()
    conn.close()

API_KEY = "SENSORED_SENSORED_SENSORED"

def get_weather(city):
    url = f"https://api.weatherapi.com/v1/forecast.json?key={API_KEY}&q={city}&days=3&aqi=no&alerts=no"
    data = requests.get(url).json()

    # Jika kota tidak ditemukan
    if "forecast" not in data:
        return None

    forecast = data["forecast"]["forecastday"]

    hasil = []
    for item in forecast:
        tanggal = item["date"]  # format 2025-11-14
        date_obj = datetime.datetime.strptime(tanggal, "%Y-%m-%d")

        hasil.append({
            "hari": date_obj.strftime("%A"),         # Senin, Selasa, ...
            "siang": item["day"]["maxtemp_c"],       # suhu siang
            "malam": item["day"]["mintemp_c"]        # suhu malam
        })

    return hasil

# panggil fungsi inisialisasi database
init_db()

@app.route("/", methods=["GET", "POST"])
def home():
    weather_data = None
    city_name = "Jakarta"  # default kota

    if request.method == "POST":
        city_name = request.form.get("city")

    # Cari koordinat kota
    geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city_name}&count=1"
    geo_response = requests.get(geo_url).json()

    if "results" not in geo_response:
        return render_template("index.html", error="Kota tidak ditemukan", weather=None)

    lat = geo_response["results"][0]["latitude"]
    lon = geo_response["results"][0]["longitude"]

    # Ambil cuaca 3 hari
    weather_url = (
        "https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}"
        "&daily=temperature_2m_max,temperature_2m_min"
        "&timezone=Asia/Jakarta"
    )

    weather_response = requests.get(weather_url).json()

    dates = weather_response["daily"]["time"]
    temp_max = weather_response["daily"]["temperature_2m_max"]
    temp_min = weather_response["daily"]["temperature_2m_min"]

    weather_data = []
    for i in range(3):
        dt = datetime.strptime(dates[i], "%Y-%m-%d")
        day_name = calendar.day_name[dt.weekday()]

        weather_data.append({
            "date": dates[i],
            "day": day_name,
            "temp_day": temp_max[i],
            "temp_night": temp_min[i]
        })

    return render_template("index.html", weather=weather_data, city=city_name)

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        confirm = request.form["confirm"]
        nickname = request.form["nickname"]

        # validasi password sama
        if password != confirm:
            return render_template("register.html",
                                   error="Password dan Konfirmasi Password tidak sama!")

        conn = get_db()

        # cek apakah username sudah ada
        existing_user = conn.execute("SELECT * FROM users WHERE username=?",
                                     (username,)).fetchone()
        if existing_user:
            return render_template("register.html",
                                   error="Username sudah digunakan!")

        # cek apakah nickname sudah ada
        existing_nick = conn.execute("SELECT * FROM users WHERE nickname=?",
                                     (nickname,)).fetchone()
        if existing_nick:
            return render_template("register.html",
                                   error="Nickname sudah digunakan!")

        # jika semua valid → simpan
        conn.execute("INSERT INTO users (username, password, nickname) VALUES (?, ?, ?)",
                     (username, password, nickname))
        conn.commit()

        return redirect("/login")

    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        conn = get_db()
        user = conn.execute("SELECT * FROM users WHERE username=? AND password=?",
                            (username, password)).fetchone()

        if user:
            session["user_id"] = user["id"]
            session["nickname"] = user["nickname"]

            # Inisialisasi score total pemain sejak registrasi (dari DB leaderboard)
            total_score = conn.execute(
                "SELECT SUM(score) FROM leaderboard WHERE user_id=?",
                (user["id"],)
            ).fetchone()[0]

            session["total_score"] = total_score if total_score else 0

            return redirect("/")
        else:
            return render_template("login.html",
                                   error="Username atau password salah!")

    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")

@app.route("/quiz", methods=["GET", "POST"])
def quiz():
    if "user_id" not in session:
        return redirect("/login")

    conn = get_db()

    # ini sialisasi skor dan jumlah soal
    if "score" not in session:
        session["score"] = 0

    if "question_count" not in session:
        session["question_count"] = 0

    # POST (menilai jawaban)
    if request.method == "POST":
        question_id = session.get("current_question_id")
        question = conn.execute("SELECT * FROM questions WHERE id=?", (question_id,)).fetchone()

        user_answer = request.form["answer"]
        correct_answer = question["correct"]

        # tambahkan jumlah soal
        session["question_count"] += 1

        if user_answer == correct_answer:
            result = "Jawaban kamu benar!"
            color = "green"
            session["score"] += 1     # skor bertambah
        else:
            result = f"Jawaban salah! Jawaban yang benar adalah {correct_answer}"
            color = "red"

        return render_template("quiz.html", 
                       q=question, 
                       result=result, 
                       color=color,
                       score=session["score"], 
                       total=session["question_count"], 
                       total_score=session["total_score"])

    if session["question_count"] >= 10:
        conn.execute("INSERT INTO leaderboard (user_id, nickname, score) VALUES (?, ?, ?)",
                    (session["user_id"], session["nickname"], session["score"]))
        conn.commit()

        # reset skor untuk permainan berikutnya
        final_score = session["score"]
        session.pop("score", None)
        session.pop("question_count", None)

        total_score = conn.execute(
                "SELECT SUM(score) FROM leaderboard WHERE user_id=?",
                (session["user_id"],)
            ).fetchone()[0]

        session['total_score'] = total_score

        return redirect(f"/finish?score={final_score}")

    # GET (ambil soal baru)
    questions = conn.execute("SELECT * FROM questions").fetchall()
    question = random.choice(questions)

    # Inisialisasi skor jika belum ada
    if "score" not in session:
        session["score"] = 0
    if "question_count" not in session:
        session["question_count"] = 0

    # Simpan ID soal ke session, supaya POST menilai soal yang sama
    session["current_question_id"] = question["id"]

    return render_template("quiz.html", 
                       q=question, 
                       result=None, 
                       color=None,
                       score=session["score"], 
                       total=session["question_count"]+1, 
                       total_score=session["total_score"])

@app.route("/finish")
def finish():
    score = request.args.get("score")
    return render_template("finish.html", score=score)

@app.route("/leaderboard")
def leaderboard():
    conn = get_db()
    data = conn.execute("""
        SELECT nickname, SUM(score) AS total_score
        FROM leaderboard
        GROUP BY user_id
        ORDER BY total_score DESC
    """).fetchall()

    return render_template("leaderboard.html", data=data)

if __name__ == "__main__":
    app.run(debug=True)
