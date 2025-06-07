import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

LR_MODEL_PATH = "diabete/diabetes_lr.joblib"
DT_MODEL_PATH = "diabete/diabetes_dt.joblib"
RF_MODEL_PATH = "diabete/diabetes_rf.joblib"

def load_data():
    data = pd.read_csv("diabete/diabete.csv")
    X = data.drop("Outcome", axis=1)
    y = data["Outcome"]
    return X, y

def train_model(model_type):
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
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
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    joblib.dump({'model': model, 'accuracy': acc}, path)
    return {'model': model, 'accuracy': acc}
def diabetes_predict(input_data):
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
        
        model = model_data['model']
        pred = model.predict(df)[0]
        proba = model.predict_proba(df)[0][1]
        acc = model_data['accuracy']
        res = f"{model_name}模型（准确率{acc:.2%}）: 预测结果{'糖尿病' if pred == 1 else '健康'}, 患病概率{proba:.2%}"
        results[f"{model_name}"] = res
    return results

# 示例用法
if __name__ == "__main__":
    # 字典格式测试数据
    sample_data = {
        "Pregnancies": 6, "Glucose": 148, "BloodPressure": 72,
        "SkinThickness": 35, "Insulin": 0, "BMI": 33.6,
        "DiabetesPedigreeFunction": 0.627, "Age": 50
    }
    # 调用新预测函数获取多模型结果
    result = diabetes_predict(sample_data)
    print("糖尿病多模型预测结果:\n",result)
