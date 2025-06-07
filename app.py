from flask import Flask, request, jsonify
from flask_cors import CORS
# 导入三个预测函数
from cancer.lung_cancer import lung_cancer_predict
from heart.heart_prediction import heart_predict
from diabete.diabetes_prediction import diabetes_predict

app = Flask(__name__)
CORS(app)

# 原心脏病接口（假设需要保留或修正）
@app.route('/api/heart', methods=['POST'])
def get_heart_disease():
    input_data = request.get_json()  # 获取前端传入的JSON数据
    result = heart_predict(input_data)
    return jsonify(result)  # 统一返回JSON格式

# 新增肺癌预测接口
@app.route('/api/cancer', methods=['POST'])
def get_lung_cancer():
    input_data = request.get_json()
    result = lung_cancer_predict(input_data)
    return jsonify(result)

# 新增糖尿病预测接口
@app.route('/api/diabetes', methods=['POST'])
def get_diabetes():
    input_data = request.get_json()
    result = diabetes_predict(input_data)
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
