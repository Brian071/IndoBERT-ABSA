pip install deep-translator
pip install langdetect
pip install Sastrawi
pip install torch 
pip install transformers 
pip install pandas nltk
pip install nltk
pip install emoji

import torch
from transformers import BertTokenizer
import pandas as pd
import re
import nltk
import nltk
import torch
import emoji


from deep_translator import GoogleTranslator
from langdetect import detect
from torch.utils.data import Dataset
from transformers import pipeline
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

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
def clean_text(text):
    # 1. Hapus semua emoji Unicode (termasuk emoticon, simbol, bendera)
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F700-\U0001F77F"  # alchemical symbols
        "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
        "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
        "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        "\U0001FA00-\U0001FA6F"  # Chess Symbols
        "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        "\U00002702-\U000027B0"  # Dingbats
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE,
    )
    text = emoji_pattern.sub("", text)

    # 2. Hapus emoji dalam format :emoji_name: 
    text = re.sub(r":(.*?):", lambda x: x.group(1).replace("_", " "), text)

    # 3. Hapus enter dan spasi berlebih
    text = text.replace("\n", " ").replace("\r", " ")
    text = " ".join(text.split())  # Normalisasi spasi

    # 4. Hapus karakter khusus, tapi pertahankan tanda baca dasar
    text = re.sub(r'[^\w\s.,!?\'"-]', "", text)

    return text.strip()

df["Review_Clean"] = df["Review_Bahasa"].apply(clean_text)

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
