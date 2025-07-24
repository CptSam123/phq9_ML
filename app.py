from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

# Load model and label map
model = joblib.load('depression_model.pkl')
label_map = joblib.load('label_map.pkl')

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return 'PHQ-9 Severity Prediction API is live 🎯'

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        answers = data.get('answers')

        if not answers or len(answers) != 9:
            return jsonify({'error': 'Expected 9 question answers'}), 400

        prediction = model.predict([answers])[0]
        severity = label_map[prediction]

        return jsonify({
            'severity': severity,
            'input': answers
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
