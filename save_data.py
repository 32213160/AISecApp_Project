# save_data.py

import requests
import time
import sqlite3
from datetime import datetime

DB_PATH = "sensor_logs.db"

def init_db():
    """
    데이터베이스 초기화 함수
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    
    cur.execute("""
    CREATE TABLE IF NOT EXISTS logs (
        timestamp TEXT,
        cpu_temp REAL,
        fan_speed REAL
    )
    """)
    
    conn.commit()
    conn.close()
    
    print(f"데이터베이스 초기화 완료: {DB_PATH}")

def get_sensor_data():
    """
    센서 데이터 수집 함수
    
    Returns:
        tuple: (cpu_temp, fan_speed) - CPU 온도와 팬 속도
    """
    try:
        # 실제 환경에서는 하드웨어 센서에서 데이터를 읽어옴
        # 여기서는 예시로 임의의 데이터 생성
        import random
        
        # 45-55도 사이의 CPU 온도 (가끔 비정상적인 값 생성)
        if random.random() < 0.05:  # 5% 확률로 비정상 온도
            cpu_temp = random.uniform(80, 95)
        else:
            cpu_temp = random.uniform(45, 55)
            
        # 1000-2000 RPM 사이의 팬 속도 (가끔 비정상적인 값 생성)
        if random.random() < 0.05:  # 5% 확률로 비정상 속도
            fan_speed = random.choice([random.uniform(0, 300), random.uniform(4000, 5000)])
        else:
            fan_speed = random.uniform(1000, 2000)
            
        return cpu_temp, fan_speed
        
    except Exception as e:
        print(f"센서 데이터 수집 중 오류 발생: {e}")
        return None, None

def save_to_db(cpu_temp, fan_speed):
    """
    데이터베이스에 센서 데이터 저장
    
    Args:
        cpu_temp (float): CPU 온도
        fan_speed (float): 팬 속도
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    
    timestamp = datetime.now().isoformat()
    cur.execute("INSERT INTO logs VALUES (?, ?, ?)", 
               (timestamp, cpu_temp, fan_speed))
    
    conn.commit()
    conn.close()
