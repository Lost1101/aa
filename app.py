from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import difflib
import re

app = Flask(__name__)
CORS(app)  # Ini akan mengizinkan semua asal (origins) untuk mengakses server ini

# Simpan percakapan dalam list
conversations = []

faq = {
    "Halo": "Halo, saya adalah Pizza-chan~ asisten Pizza Place anda, apakah ada yang bisa saya bantu?",
    "Bisakah anda jelaskan tentang chart total sales per year?": "Total sales untuk tahun ini naik dan turun perbulannya, sehingga tidak stabil dan sulit untuk diprediksi, maka ini menjadi masalah utama untuk toko pizza kita :(",
    "Bisakah anda jelaskan tentang chart busy time?": "Pizza terjual laris saat sore hari, kemungkinan dikarenakan jam kerja selesai pada waktu tersebut, dan waktu paling sedikit pembeli ialah malam hari, dikarenakan orang-orang sudah tidur nyenyak :D",
    "Bagaimana cara menghubungi dukungan pelanggan?": "Anda dapat menghubungi dukungan pelanggan melalui email di support@example.com atau melalui telepon di 123-456-7890."
}

questions = list(faq.keys())
answers = list(faq.values())

# Fit the TF-IDF vectorizer on the FAQ questions
vectorizer = TfidfVectorizer().fit(questions)
faq_vectors = vectorizer.transform(questions)

def normalize_text(text):
    # Lowercase the text and remove non-alphanumeric characters
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    return text

def levenshtein_distance(str1, str2):
    return difflib.SequenceMatcher(None, str1, str2).ratio()

def find_best_match(user_input, faq_list, threshold=0.4):
    user_input = normalize_text(user_input)
    user_input_vector = vectorizer.transform([user_input])
    cosine_similarities = cosine_similarity(user_input_vector, faq_vectors)
    max_similarity = cosine_similarities.max()

    if max_similarity >= threshold:
        best_match_index = cosine_similarities.argmax()
        best_match_question = questions[best_match_index]

        # Apply Levenshtein distance as a secondary check
        levenshtein_ratio = levenshtein_distance(user_input, normalize_text(best_match_question))

        if levenshtein_ratio >= threshold:
            return best_match_question, levenshtein_ratio

    return None, 0

@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    user_input = data['question']
    best_match, ratio = find_best_match(user_input, list(faq.keys()), threshold=0.4)  # Lower the threshold to 0.4
    if best_match:
        response = faq[best_match]
    else:
        response = "Maaf, saya tidak mengerti pertanyaan Anda."

    # Simpan pertanyaan dan jawaban dalam percakapan
    conversations.append({'question': user_input, 'answer': response})

    return jsonify({"answer": response, "conversations": conversations})

if __name__ == '__main__':
    app.run(debug=True)