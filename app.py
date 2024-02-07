from flask import Flask, render_template, request, jsonify
from qa_model import get_qa_model
import torch
app = Flask(__name__)

#1. model_path 로 설정하기
model_path = 'model/checkpoint-1119-epoch-1'
qa_model = get_qa_model(model_path, use_cuda=False)

#2. pytorch pt 파일로 설정하기
#model_path = 'model/checkpoint-1119-epoch-1/optimizer.pt'
#qa_model = get_ga_model()
#qa_model.load_state_dict(torch.load(PATH, map_location=device), strict = False)

#model.eval()
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/qa', methods=['POST'])
def qa_endpoint():
    print(1)
    try:
        print(2)
        data = request.get_json(force=True)
        context = data['context']
        question = data['question']

        to_predict = [{"context": context, "qas": [{"question": question, "id": "0"}]}]
        result = qa_model.predict(to_predict)

        answer = result[0]['answer'][0]

        response = {'answer': answer}
        
    except Exception as e:
        print(3)
        response = {'error': str(e)}

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)