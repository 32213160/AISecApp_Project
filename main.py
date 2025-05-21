# main.py

import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from preprocess import load_data, extract_features
from model import train_model, predict
from save_data import init_db, get_sensor_data, save_to_db

def collect_data(duration_minutes=10, interval_seconds=5):
    """
    지정된 시간 동안 센서 데이터를 수집하여 데이터베이스에 저장합니다.
    
    Args:
        duration_minutes (int): 데이터 수집 기간(분)
        interval_seconds (int): 데이터 수집 간격(초)
    """
    print(f"{duration_minutes}분 동안 {interval_seconds}초 간격으로 데이터를 수집합니다.")
    
    # DB 초기화
    init_db()
    
    end_time = time.time() + (duration_minutes * 60)
    
    try:
        while time.time() < end_time:
            # 센서 데이터 수집
            cpu_temp, fan_speed = get_sensor_data()
            
            if cpu_temp is not None and fan_speed is not None:
                # DB에 저장
                save_to_db(cpu_temp, fan_speed)
                print(f"저장됨: CPU 온도 {cpu_temp}°C, 팬 속도 {fan_speed} RPM")
            
            time.sleep(interval_seconds)
            
        print("데이터 수집이 완료되었습니다.")
    
    except KeyboardInterrupt:
        print("\n프로그램이 사용자에 의해 종료되었습니다.")

def visualize_data():
    """
    수집된 데이터를 시각화합니다.
    """
    # 데이터 로드
    df = load_data()
    
    if df.empty:
        print("시각화할 데이터가 없습니다.")
        return
    
    # 시각화
    plt.figure(figsize=(12, 8))
    
    # CPU 온도 그래프
    plt.subplot(2, 1, 1)
    plt.plot(df['timestamp'], df['cpu_temp'], 'r-', label='CPU 온도')
    plt.title('CPU 온도 변화')
    plt.ylabel('온도 (°C)')
    plt.legend()
    plt.grid(True)
    
    # 팬 속도 그래프
    plt.subplot(2, 1, 2)
    plt.plot(df['timestamp'], df['fan_speed'], 'b-', label='팬 속도')
    plt.title('팬 속도 변화')
    plt.xlabel('시간')
    plt.ylabel('속도 (RPM)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('hardware_monitoring.png')
    plt.show()
    
    print("데이터 시각화가 완료되었습니다.")

def anomaly_detection():
    """
    이상 탐지 모델을 학습하고 결과를 시각화합니다.
    """
    # 데이터 로드 및 전처리
    df = load_data()
    
    if df.empty:
        print("분석할 데이터가 없습니다.")
        return
    
    X = extract_features(df)
    
    # 모델 학습
    model = train_model(X)
    
    # 예측
    predictions = predict(model, X)
    
    # 결과 시각화
    plt.figure(figsize=(12, 8))
    
    # 이상치 표시
    normal = predictions == 1
    anomaly = predictions == -1
    
    plt.subplot(2, 1, 1)
    plt.plot(df['timestamp'], df['cpu_temp'], 'b-', label='CPU 온도')
    plt.scatter(df.loc[normal, 'timestamp'], df.loc[normal, 'cpu_temp'], c='green', label='정상', alpha=0.5)
    plt.scatter(df.loc[anomaly, 'timestamp'], df.loc[anomaly, 'cpu_temp'], c='red', label='이상', alpha=0.5)
    plt.title('CPU 온도 이상 탐지')
    plt.ylabel('온도 (°C)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(df['timestamp'], df['fan_speed'], 'b-', label='팬 속도')
    plt.scatter(df.loc[normal, 'timestamp'], df.loc[normal, 'fan_speed'], c='green', label='정상', alpha=0.5)
    plt.scatter(df.loc[anomaly, 'timestamp'], df.loc[anomaly, 'fan_speed'], c='red', label='이상', alpha=0.5)
    plt.title('팬 속도 이상 탐지')
    plt.xlabel('시간')
    plt.ylabel('속도 (RPM)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('anomaly_detection.png')
    plt.show()
    
    # 이상치 비율 출력
    anomaly_ratio = np.mean(predictions == -1) * 100
    print(f"이상치 비율: {anomaly_ratio:.2f}%")
    print(f"총 {len(predictions)}개의 데이터 중 {np.sum(predictions == -1)}개의 이상치가 발견되었습니다.")

def real_time_monitoring(duration_minutes=5, interval_seconds=1):
    """
    실시간으로 데이터를 수집하고 이상 탐지를 수행합니다.
    
    Args:
        duration_minutes (int): 모니터링 기간(분)
        interval_seconds (int): 데이터 수집 간격(초)
    """
    # 기존 데이터로 모델 학습
    df = load_data()
    
    if df.empty:
        print("학습할 데이터가 없습니다. 먼저 collect_data()를 실행하세요.")
        return
    
    X = extract_features(df)
    model = train_model(X)
    
    # 실시간 모니터링 설정
    plt.ion()  # 대화형 모드 활성화
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # 데이터 저장용 리스트
    timestamps = []
    cpu_temps = []
    fan_speeds = []
    statuses = []
    
    end_time = time.time() + (duration_minutes * 60)
    
    try:
        while time.time() < end_time:
            # 센서 데이터 수집
            cpu_temp, fan_speed = get_sensor_data()
            
            if cpu_temp is not None and fan_speed is not None:
                # DB에 저장
                save_to_db(cpu_temp, fan_speed)
                
                # 데이터 추가
                timestamp = datetime.now()
                timestamps.append(timestamp)
                cpu_temps.append(cpu_temp)
                fan_speeds.append(fan_speed)
                
                # 이상 탐지
                features = pd.DataFrame({
                    'cpu_temp': [cpu_temp],
                    'fan_speed': [fan_speed],
                    'cpu_temp_ma': [cpu_temp],  # 단일 데이터이므로 이동평균 대신 현재값 사용
                    'fan_speed_ma': [fan_speed]
                })
                
                prediction = predict(model, features)[0]
                status = "정상" if prediction == 1 else "이상"
                statuses.append(1 if status == "정상" else 0)
                
                # 최근 30개 데이터만 표시
                if len(timestamps) > 30:
                    timestamps.pop(0)
                    cpu_temps.pop(0)
                    fan_speeds.pop(0)
                    statuses.pop(0)
                
                # 그래프 업데이트
                ax1.clear()
                ax2.clear()
                
                # CPU 온도 그래프
                ax1.plot(range(len(timestamps)), cpu_temps, 'r-')
                ax1.set_ylabel('CPU 온도 (°C)')
                ax1.set_title('실시간 CPU 온도 모니터링')
                ax1.grid(True)
                
                # 팬 속도 그래프
                ax2.plot(range(len(timestamps)), fan_speeds, 'b-')
                ax2.set_ylabel('팬 속도 (RPM)')
                ax2.set_title('실시간 팬 속도 모니터링')
                ax2.grid(True)
                
                # 상태 표시
                status_color = "green" if status == "정상" else "red"
                plt.figtext(0.5, 0.01, f"현재 상태: {status}", ha="center", fontsize=14, 
                           bbox={"facecolor": status_color, "alpha": 0.2})
                
                plt.tight_layout()
                plt.pause(interval_seconds)
                
                print(f"[{timestamp.strftime('%Y-%m-%d %H:%M:%S')}] CPU: {cpu_temp}°C, 팬: {fan_speed} RPM, 상태: {status}")
            
            time.sleep(interval_seconds)
        
        plt.ioff()
        print("실시간 모니터링이 완료되었습니다.")
    
    except KeyboardInterrupt:
        plt.ioff()
        print("\n실시간 모니터링이 사용자에 의해 종료되었습니다.")

def main():
    """
    메인 함수 - 사용자 인터페이스 제공
    """
    print("=== 하드웨어 모니터링 및 이상 탐지 시스템 ===")
    print("1. 데이터 수집")
    print("2. 데이터 시각화")
    print("3. 이상 탐지 분석")
    print("4. 실시간 모니터링")
    print("0. 종료")
    
    choice = input("원하는 기능을 선택하세요: ")
    
    if choice == '1':
        duration = int(input("수집 기간(분)을 입력하세요: "))
        interval = int(input("수집 간격(초)을 입력하세요: "))
        collect_data(duration, interval)
    elif choice == '2':
        visualize_data()
    elif choice == '3':
        anomaly_detection()
    elif choice == '4':
        duration = int(input("모니터링 기간(분)을 입력하세요: "))
        interval = int(input("수집 간격(초)을 입력하세요: "))
        real_time_monitoring(duration, interval)
    elif choice == '0':
        print("프로그램을 종료합니다.")
        return
    else:
        print("잘못된 선택입니다.")
    
    # 재귀적으로 메인 메뉴 호출
    main()

if __name__ == "__main__":
    main()
