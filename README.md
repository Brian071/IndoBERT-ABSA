# Analisis Sentimen Berbasis Aspek (ABSA) dengan IndoBERT

Repositori ini berisi kode dan sumber daya untuk melakukan Analisis Sentimen Berbasis Aspek pada teks berbahasa Indonesia menggunakan model IndoBERT.

## Deskripsi Proyek

Proyek ini bertujuan untuk mengklasifikasikan sentimen (positif, negatif) dan aspek-aspek spesifik dalam review Hotel (Kamar, Staff, Layanan, Hotel, Kolam Renang, Harga, Sarapan, Fasilitas & Lokasi

## Struktur Folder

Proyek ini mengikuti struktur direktori standar untuk proyek machine learning
ASBA-Project/
├── README.md
├── requirements.txt
├── config.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── aspect_labels.csv
├── notebooks/
│   ├── eda.ipynb
│   ├── preprocessing.ipynb
│   └── modeling.ipynb
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── model.py
│   └── utils.py
├── outputs/
│   ├── metrics.txt
│   └── plots/
├── .gitignore
└── LICENSE
