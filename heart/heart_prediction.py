from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import os

LR_MODEL_PATH = 'heart/heart_lr.pkl'
DT_MODEL_PATH = 'heart/heart_dt.pkl'
RF_MODEL_PATH = 'heart/heart_rf.pkl'
def load_data():
    # 新增数据加载函数
    data = pd.read_csv('heart/heart.csv')
    X = data.drop('target', axis=1)
    y = data['target']
    return X, y

def train_model(model_type):
    # 修改训练函数按模型类型训练并保存
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    if model_type == '逻辑回归':
        model = LogisticRegression(max_iter=1000, random_state=42)
        path = LR_MODEL_PATH
    elif model_type == '决策树':
        model = DecisionTreeClassifier(max_depth=3, random_state=42)
        path = DT_MODEL_PATH
    elif model_type == '随机森林':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        path = RF_MODEL_PATH
    else:
        raise ValueError("未知模型类型")
    
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    joblib.dump({'model': model, 'scaler': scaler, 'accuracy': acc}, path)
    return {'model': model, 'scaler': scaler, 'accuracy': acc}  # 返回当前模型数据

def heart_predict(input_data):
    df = pd.DataFrame([input_data])
    results = {}
    model_types = ['逻辑回归', '决策树', '随机森林']
    for model_name in model_types:
        if model_name == '逻辑回归':
            path = LR_MODEL_PATH
        elif model_name == '决策树':
            path = DT_MODEL_PATH
        else:
            path = RF_MODEL_PATH
        
        if os.path.exists(path):
            model_data = joblib.load(path)
        else:
            model_data = train_model(model_name)
        
        scaler = model_data['scaler']
        input_scaled = scaler.transform(df)
        model = model_data['model']
        pred = model.predict(input_scaled)[0]
        proba = model.predict_proba(input_scaled)[0][1]
        acc = model_data['accuracy']
        res = f"{model_name}模型（准确率{acc:.2%}）: 预测结果{'心脏病' if pred == 1 else '健康'}, 患病概率{proba:.2%}"
        results[f"{model_name}"] = res
    return results

if __name__ == "__main__":
    # 字典格式测试数据
    sample_data = {
        'age': 63, 'sex': 1, 'cp': 3, 'trestbps': 145,
        'chol': 233, 'fbs': 1, 'restecg': 0, 'thalach': 150,
        'exang': 0, 'oldpeak': 2.3, 'slope': 0, 'ca': 0,
        'thal': 1
    }
    # 调用新预测函数获取多模型结果
    result = heart_predict(sample_data)
    print("心脏病多模型预测结果:\n",result)
    