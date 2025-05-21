# preprocess.py

import pandas as pd
import sqlite3

def load_data():
    """
    데이터베이스에서 센서 데이터 로드
    
    Returns:
        DataFrame: 센서 데이터
    """
    conn = sqlite3.connect("sensor_logs.db")
    
    try:
        df = pd.read_sql_query("SELECT * FROM logs", conn)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.dropna()
        
        print(f"데이터 로드 완료: {df.shape[0]}개의 데이터")
        return df
    
    except Exception as e:
        print(f"데이터 로드 중 오류 발생: {e}")
        return pd.DataFrame()
    
    finally:
        conn.close()

def extract_features(df):
    """
    특성 추출 함수
    
    Args:
        df (DataFrame): 원본 데이터
        
    Returns:
        DataFrame: 추출된 특성
    """
    # 이동 평균 계산 (노이즈 감소 효과)
    df["cpu_temp_ma"] = df["cpu_temp"].rolling(window=3).mean()
    df["fan_speed_ma"] = df["fan_speed"].rolling(window=3).mean()
    
    # 결측치 제거
    df = df.dropna()
    
    # 필요한 특성만 선택
    features = df[["cpu_temp", "fan_speed", "cpu_temp_ma", "fan_speed_ma"]]
    
    print(f"특성 추출 완료: {features.shape[1]}개의 특성")
    return features
