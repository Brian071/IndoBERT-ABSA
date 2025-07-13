pip install deep-translator
pip install langdetect
pip install Sastrawi
pip install torch 
pip install transformers 
pip install pandas nltk
pip install nltk

import torch
from transformers import BertTokenizer
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import nltk
from deep_translator import GoogleTranslator
from langdetect import detect
import nltk
import torch
from torch.utils.data import Dataset
from transformers import pipeline

nltk.download('punkt')
nltk.download('stopwords')

from google.colab import files
uploaded = files.upload() 
import pandas as pd
df = pd.read_csv('Bali_Hotel_Review.csv', sep=';')

# Set opsi tampilan untuk menampilkan semua kolom
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)  # Lebar tampilan konsol

# Membaca dataset
df = pd.read_csv('Bali_Hotel_Review.csv', sep=';')

# Menghapus review yang duplikat
df.drop_duplicates(subset='Review', inplace=True)

# Menampilkan teks "Dataset sebelum preprocessing:" di konsol
print("Dataset sebelum preprocessing:")

# Cek kolom dan data
print("\nKolom dataset:", df.columns)
print("\nJumlah baris dan kolom:", df.shape)
print("\nData sampel (10 baris pertama):")
print(df.head(10))


# Function to translate text with retries
def translate_text_with_retry(text, retries=3):
    for attempt in range(retries):  # Coba beberapa kali
        try:
            translation = GoogleTranslator(source='auto', target='en').translate(text)
            return translation  # Kembalikan hasil translasi jika berhasil
        except Exception as e:
            print(f"Translation attempt {attempt + 1} failed for text: {text} -> {e}")
    print(f"All attempts failed for text: {text}. Returning original text.")
    return text  # Gunakan teks asli jika semua percobaan gagal

# Translate the 'Review' column with retries
if 'Review' in df.columns:
   df['Review_English'] = df['Review'].apply(lambda x: translate_text_with_retry(x, retries=3))
   df.to_csv('Bali_Hotel_Review.csv', index=False)
   print("Translation complete. Translated file saved as Bali_Hotel_Review.csv")
else:
    print("Error: 'Review' column not found in the CSV file.")

# Load summarization pipeline dari Hugging Face
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Function untuk meringkas teks
def summarize_review(text, max_token_length=256):
    try:
        # Hitung panjang input teks (jumlah token/word)
        input_length = len(text.split())

        # Jangan diringkas jika panjang input sudah di bawah 256 token
        if input_length < max_token_length:
            return text  # Kembalikan teks asli tanpa ringkasan

        # Ringkas teks jika panjang input melebihi 256 token
        summary = summarizer(
            text,
            max_length=max_token_length, # Maksimum panjang ringkasan
            min_length=50,               # Panjang minimum ringkasan
            do_sample=False              # Hasil deterministik
        )
        return summary[0]["summary_text"]  # Kembalikan ringkasan
    except Exception as e:
        print(f"Error summarizing text: {text} -> {e}")
        return text  # Jika gagal, kembalikan teks asli

# Pastikan Review_Bahasa sudah berisi teks panjang yang ingin diringkas
df['Review_Summary'] = df['Review_English'].apply(lambda x: summarize_review(x, max_token_length=256))

# Output hasil DataFrame dengan ringkasan
print(df)

# Cek hasil
print(df.head(10))

# Translate ke dalam bahasa Indonesia
def translate_text_with_retry(text, retries=3):
    for attempt in range(retries):  # Coba beberapa kali
        try:
            translation = GoogleTranslator(source='en', target='id').translate(text)
            return translation  # Kembalikan hasil translasi jika berhasil
        except Exception as e:
            print(f"Translation attempt {attempt + 1} failed for text: {text} -> {e}")
    print(f"All attempts failed for text: {text}. Returning original text.")
    return text  # Gunakan teks asli jika semua percobaan gagal

# Translate the 'Review' column with retries
if 'Review_English' in df.columns:
   df['Review_Bahasa'] = df['Review_English'].apply(lambda x: translate_text_with_retry(x, retries=3))
   df.to_csv('Bali_Hotel_Review.csv', index=False)
   print("Translation complete. Translated file saved as Bali_Hotel_Review.csv")
else:
    print("Error: 'Review_English' column not found in the CSV file.")

# Cek hasil
print(df.head(10))

# Fungsi preprocessing
def preprocess_text(text):
    text = text.lower()  # Lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Hapus angka & tanda baca
    # Download 'punkt_tab' resource if it hasn't been downloaded yet
    try:
        word_tokenize("test") # Try to tokenize a test string to check if resource is available
    except LookupError:
        nltk.download('punkt_tab') # Download the resource if it's not found
    tokens = word_tokenize(text)  # Tokenisasi
    stop_words = set(stopwords.words('english'))  # Stopwords English
    tokens = [word for word in tokens if word not in stop_words]  # Filter stopwords
    stemmer = PorterStemmer()  # Stemming
    tokens = [stemmer.stem(word) for word in tokens]  # Stem tiap kata
    return ' '.join(tokens)  # Gabung kembali jadi string

# REPLACE langsung kolom "Review" dengan data yang sudah diproses
df["Review_English"] = df["Review_English"].apply(preprocess_text)

# Cek hasil
print(df.head(10))

# Ekstraksi fitur teks (Review) dan label (Rating)
texts = df['Review_English'].tolist()  # Menggunakan kolom 'Review' untuk analisis teks
labels = df['Rating'].tolist()  # Menggunakan 'Rating' sebagai label target

# Konversi 'Rating' ke kategori jika diperlukan
# Nilai >3 adalah positif (1) dan <=3 adalah negatif (0)
df['Label'] = df['Rating'].apply(lambda rating: 1 if rating > 3 else 0)

print(df.head())

# Pastikan menyimpan DataFrame setelah menambahkan kolom
df.to_csv('Bali_Hotel_Review.csv', index=False) 
print(label_counts)

Bali_Hotel_Review_Clean = df[['Rating', 'Review_Clean', 'Label']].copy()

# Menampilkan data yang sudah terfilter
print("Data yang sudah terfilter:")
print(Bali_Hotel_Review_Clean.head(10)) 

# Hitung jumlah label
label_counts = Bali_Hotel_Review_Clean['Label'].value_counts()

# Tampilkan hasil
print("Jumlah label 0 dan 1:")
print(label_counts)

# Fungsi untuk membersihkan teks dan menghapus enter agar jadi paragraf
def clean_text(text):
    # Hapus enter (\n dan \r) agar teks menjadi satu paragraf utuh
    text = text.replace("\n", " ").replace("\r", " ")

    return text

# Terapkan pada kolom data teks
Bali_Hotel_Review_Clean["Review_Clean"] = Bali_Hotel_Review_Clean["Review_Clean"].apply(clean_text)

# Cek hasil
print(Bali_Hotel_Review_Clean.head(10))

# Simpan DataFrame setelah menambahkan kolom
Bali_Hotel_Review_Clean.to_csv('Bali_Hotel_Review_Clean.csv', index=False)
