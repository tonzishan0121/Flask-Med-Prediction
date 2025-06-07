import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os

LR_MODEL_PATH = "./cancer/lung_cancer_lr.joblib"
DT_MODEL_PATH = "./cancer/lung_cancer_dt.joblib"
RF_MODEL_PATH = "./cancer/lung_cancer_rf.joblib"

def load_data():
    data = pd.read_csv("./cancer/lung_cancer.csv")
    # 新增特征列名规范化处理
    data.columns = data.columns.str.strip().str.replace(' ', '_')
    # 转换二分类标签
    data["LUNG_CANCER"] = data["LUNG_CANCER"].map({"YES": 1, "NO": 0})
    X = data.drop("LUNG_CANCER", axis=1)
    y = data["LUNG_CANCER"]
    
    # 性别特征编码
    X["GENDER"] = X["GENDER"].map({"M": 1, "F": 0})
    return X, y

def train_model(model_type):
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 数据标准化
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
    joblib.dump({'model': model, 'scaler': scaler, 'feature_columns': X.columns.tolist(), 'accuracy': acc}, path)
    return {'model': model, 'scaler': scaler, 'feature_columns': X.columns.tolist(), 'accuracy': acc}

def lung_cancer_predict(input_data):
    # 预处理输入数据
    input_df = pd.DataFrame([input_data])
    input_df.columns = input_df.columns.str.strip().str.replace(' ', '_')
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
        
        # 对齐列顺序
        input_aligned = input_df[model_data['feature_columns']]
        scaler = model_data['scaler']
        scaled_data = scaler.transform(input_aligned)
        model = model_data['model']
        pred = model.predict(scaled_data)[0]
        proba = model.predict_proba(scaled_data)[0][1]
        acc = model_data['accuracy']
        res = f"{model_name}模型（准确率{acc:.2%}）: 预测结果{'肺癌阳性' if pred == 1 else '健康'}, 阳性概率{proba:.2%}"
        results[f"{model_name}"] = res
    return results



# 示例用法
if __name__ == "__main__":
    # 字典格式测试数据
    sample_data = {
        "GENDER": 1, "AGE": 65, "SMOKING": 2, "YELLOW_FINGERS": 1,
        "ANXIETY": 2, "PEER_PRESSURE": 1, "CHRONIC_DISEASE": 1,
        "FATIGUE": 2, "ALLERGY": 1, "WHEEZING": 2,
        "ALCOHOL_CONSUMING": 2, "COUGHING": 2, 
        "SHORTNESS_OF_BREATH": 2, "SWALLOWING_DIFFICULTY": 2,
        "CHEST_PAIN": 2
    }
    # 调用新预测函数获取多模型结果
    result = lung_cancer_predict(sample_data)
    print("肺癌多模型预测结果:\n",result)