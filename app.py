from flask import Flask, render_template, request, jsonify
from document import DocumentQA

app = Flask(__name__)
qa_system = DocumentQA()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    document = data.get('document', '')
    question = data.get('question', '')
    
    if not document or not question:
        return jsonify({'error': 'Both document and question are required'}), 400
    
    answer = qa_system.answer_question(question, document)
    return jsonify({'answer': answer})

if __name__ == '__main__':
    app.run(debug=True)