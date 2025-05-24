# 👁️ Blink Talk - Eye-Controlled Communication System

**Blink Talk** adalah sebuah proyek berbasis _Computer Vision_ yang memungkinkan pengguna untuk berkomunikasi menggunakan kedipan mata dan arah pandangan. Sistem ini memanfaatkan deteksi landmark wajah dan penghitungan Eye Aspect Ratio (EAR) untuk mendeteksi kedipan dan gerakan mata sebagai bentuk input.

## 🚀 Fitur

- Deteksi kedipan otomatis dengan Eye Aspect Ratio (EAR)
- Deteksi arah pandangan mata (kiri, kanan, atas, tengah)
- Pemrosesan kontur mata untuk deteksi gerakan bola mata
- Pemetaan gerakan mata menjadi teks terjemahan
- Output teks dan suara dengan `pyttsx3`
- Dukungan multi-bahasa dengan modul `translate`

## 📦 Teknologi yang Digunakan

- Python 3
- OpenCV
- Dlib (68-point facial landmark detector)
- NumPy
- pyttsx3 (text-to-speech)
- translate (terjemahan bahasa)
- Webcam (sebagai input visual)

## 🧠 Konsep Utama

### Eye Aspect Ratio (EAR)

EAR dihitung untuk mendeteksi apakah mata sedang berkedip berdasarkan proporsi antara lebar horizontal dan tinggi vertikal mata.

### Eye Movement Detection

Posisi bola mata (kiri, kanan, atas, tengah) dianalisis dengan memproses threshold dan kontur dari area mata.

## 🔧 Instalasi

1. **Clone repo ini**:

```bash
git clone https://github.com/username/blink-talk.git
cd blink-talk
```
2. **install dependensi**:

```bash
pip install -r requirements.txt
```
*Jika belum ada, pastikan Anda mendownload file shape_predictor_68_face_landmarks.dat dari:
https://github.com/davisking/dlib-models*

**Letakkan file tersebut di folder models/.**

## 🧪 Menjalankan Aplikasi
Pastikan webcam Anda aktif, kemudian jalankan:

```bash
python main.py
```

## 💬 Contoh Mapping Input

| Gerakan Mata       | Teks Output             |
| ------------------ | ----------------------- |
| left + left + left | "bey wan dekk ??"       |

## 📁 Struktur Proyek
```
blink-talk/
│
├── models/
│   └── shape_predictor_68_face_landmarks.dat
├── main.py
├── requirements.txt
└── README.md
```

