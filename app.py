from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from qa_model import get_qa_model

app = Flask(__name__)

cors = CORS(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/qa', methods=['POST'])
def qa_endpoint():
    try:
        data = request.get_json(force=True)

        context = data['context']
        question = data['question']

        to_predict = [{"context": context, "qas": [{"question": question, "id": "0"}]}]
        
        result = qa_model.predict(to_predict)

        answer = result[0][0]['answer'][0]
        answer = "적절한 답변을 찾을 수 없습니다." if answer == '' else answer

        response = jsonify({'answer': answer})
        
    except Exception as e:
        response = jsonify({'error': str(e)})

    return response

if __name__ == '__main__':
    model_path = 'model/checkpoint-1119-epoch-1'
    qa_model = get_qa_model(model_path, use_cuda=False)
    app.run(host='127.0.0.1', port=5000)
