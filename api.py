# api.py

from flask import Flask, jsonify, request
import pandas as pd
from preprocess import load_data, extract_features
from model import predict
import joblib

app = Flask(__name__)

@app.route("/status", methods=["GET"])
def get_status():
    """
    최신 시스템 상태를 반환하는 API 엔드포인트
    """
    try:
        # 모델 로드
        model = joblib.load("model.pkl")
        
        # 최신 데이터 로드
        df = load_data().tail(10)
        X = extract_features(df)
        latest = X.iloc[[-1]]
        
        # 예측
        result = predict(model, latest)[0]
        status = "정상" if result == 1 else "이상"
        
        return jsonify({
            "cpu_temp": latest["cpu_temp"].values[0],
            "fan_speed": latest["fan_speed"].values[0],
            "status": status,
            "timestamp": df.iloc[-1]["timestamp"].isoformat()
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(port=5000, debug=True)
