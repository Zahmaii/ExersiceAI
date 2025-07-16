import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import datetime
import time
import tempfile
import os
import cv2

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from matplotlib.colors import LinearSegmentedColormap
from streamlit_option_menu import option_menu

selected = option_menu(
    menu_title="Exercise AI Tools | 운동 AI 도구.",  
    options=["AI Detection | AI 탐지", "AI Prediction | AI 예측"],  
    icons=["camera", "bar-chart-steps"],  
    menu_icon="cast",
    default_index=0,  
    orientation="horizontol"
)

if selected == "AI Detection | AI 탐지":
    # Constants
    CSV_FILE = "exercise_history.csv"

    # Initialize CSV if not exist
    if "initialized" not in st.session_state:
        try:
            pd.read_csv(CSV_FILE) 
        except FileNotFoundError:
            df_init = pd.DataFrame(columns=["Date | 날짜", "Exercise | 운동", "Repetition | 반복"])
            df_init.to_csv(CSV_FILE, index=False)
        st.session_state.initialized = True

    st.title("🏋 Exercise Repetition Detection | 운동반복탐지")

    # Sidebar
    exercise_type = st.sidebar.selectbox(
        "Choose Exercise | 연습 선택",
        ["Choose... | 선택한다...", "Squat | 스쿼트", "Push Up | 밀어올리다", "Lunges | 런지", "Jumping Jacks | 점프잭"]
    )

    enable_timer = st.sidebar.checkbox("⏲ Timer Activation | 타이머 활성화")
    if enable_timer:
        duration = st.sidebar.number_input("Timer Settings (Seconds) | 타이머 설정(초)", min_value=10, value=30)

    stop_session = st.sidebar.button("⛔ Stop The Session | 세션 중지")


    # Session state
    if "run_camera" not in st.session_state:
        st.session_state.run_camera = False
    if "counter" not in st.session_state:
        st.session_state.counter = 0
    if "stage" not in st.session_state:
        st.session_state.stage = None
    if "start_time" not in st.session_state:
        st.session_state.start_time = None

    # Angle calculation
    def calculate_angle(a, b, c):
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)

        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)

        if angle > 180.0:
            angle = 360 - angle

        return angle

    # MediaPipe setup
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    # Only run camera if exercise selected
    if exercise_type != "Choose... | 선택한다...":
        # st.info(f"Start performing *{exercise_type}* in front of your webcam.")

        start_button = st.button("▶ Turn On The Camera | 카메라를 켜세요")

        if start_button:
            st.session_state.run_camera = True
            st.session_state.counter = 0
            st.session_state.stage = None
            st.session_state.start_time = time.time()

        FRAME_WINDOW = st.image([])

        if st.session_state.run_camera and not stop_session:
            cap = cv2.VideoCapture(0)

            with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
                while st.session_state.run_camera and not stop_session:
                    ret, frame = cap.read()
                    if not ret:
                        st.warning("I can't access the camera | 카메라에 접근할 수 없습니다.")
                        break

                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image.flags.writeable = False
                    results = pose.process(image)
                    image.flags.writeable = True
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                    try:
                        landmarks = results.pose_landmarks.landmark

                        if exercise_type == "Squat | 스쿼트":
                            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                            knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                                    landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                            ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                                    landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                            angle = calculate_angle(hip, knee, ankle)
                            cv2.putText(image, str(int(angle)),
                                        tuple(np.multiply(knee, [640, 480]).astype(int)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                            if angle > 160:
                                st.session_state.stage = "up"
                            if angle < 120 and st.session_state.stage == 'up':
                                st.session_state.stage = "down"
                                st.session_state.counter += 1

                        elif exercise_type == "Push Up | 밀어올리다":
                            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                                    landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                                    landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                            angle = calculate_angle(shoulder, elbow, wrist)
                            cv2.putText(image, str(int(angle)),
                                        tuple(np.multiply(elbow, [640, 480]).astype(int)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                            if angle > 160:
                                st.session_state.stage = "up"
                            if angle < 90 and st.session_state.stage == 'up':
                                st.session_state.stage = "down"
                                st.session_state.counter += 1
                                
                        elif exercise_type == "Lunges | 런지":
                            # Uses hip-knee-ankle angle for front leg
                            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                            knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                                    landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                            ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                                    landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                            
                            angle = calculate_angle(hip, knee, ankle)
                            cv2.putText(image, str(int(angle)),
                                        tuple(np.multiply(knee, [640, 480]).astype(int)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                            
                            # Lunge detection logic
                            if angle > 160:  # Standing position
                                st.session_state.stage = "up"
                            if angle < 120 and st.session_state.stage == 'up':  # Lunge down
                                st.session_state.stage = "down"
                                st.session_state.counter += 1

                        elif exercise_type == "Jumping Jacks | 점프잭":
                            # Uses shoulder-elbow-wrist angle for arm movement
                            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                                    landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                                    landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                            
                            # Calculate angle between shoulder-elbow-wrist
                            angle = calculate_angle(shoulder, elbow, wrist)
                            cv2.putText(image, str(int(angle)),
                                        tuple(np.multiply(elbow, [640, 480]).astype(int)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                            
                            # Also check wrist height relative to shoulder for arm position
                            wrist_height = wrist[1]
                            shoulder_height = shoulder[1]
                            
                            # Jumping jack detection logic
                            if wrist_height > shoulder_height:  # Arms down
                                st.session_state.stage = "down"
                            if wrist_height < shoulder_height and st.session_state.stage == 'down':  # Arms up
                                st.session_state.stage = "up"
                                st.session_state.counter += 1

                        mp_drawing.draw_landmarks(
                            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2),
                            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2)
                        )

                    except:
                        pass

                    # Timer logic
                    if enable_timer:
                        elapsed = time.time() - st.session_state.start_time
                        remaining = int(duration - elapsed)
                        cv2.putText(image, f"Timer Left: {remaining}s", (380, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                        if remaining <= 0:
                            stop_session = True

                    # Draw counter
                    cv2.rectangle(image, (0, 0), (250, 80), (245, 117, 16), -1)
                    cv2.putText(image, 'Reps', (15, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
                    cv2.putText(image, str(st.session_state.counter), (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)

                    FRAME_WINDOW.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

                    if stop_session:
                        break

                cap.release()
                cv2.destroyAllWindows()
                st.session_state.run_camera = False

        # Save to CSV
        if stop_session and exercise_type != "Choose... | 선택한다...":
            st.session_state.run_camera = False
            st.success(f"End of Session. Total Reps: {st.session_state.counter} | 세션 종료. 만차 반복: {st.session_state.counter}")

            # Append to CSV
            df_new = pd.DataFrame({
                "Date | 날짜": [datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
                "Exercise | 운동": [exercise_type],
                "Repetition | 반복": [st.session_state.counter]
            })
            df_existing = pd.read_csv(CSV_FILE)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            df_combined.to_csv(CSV_FILE, index=False)

            st.write("✅ Session saved to history. | 세션이 기록에 저장되었습니다.")

            # Show past history for current exercise
            st.subheader(f"📈 Historical Records for {exercise_type} | 역사적 기록 {exercise_type}")
            filtered = df_combined[df_combined["Exercise | 운동"] == exercise_type]
            st.dataframe(filtered.tail(10), use_container_width=True)
    else:
        st.warning("👈 Please select an exercise to begin. | 시작할 연습 문제를 선택해 주세요.")
        
elif selected == "AI Prediction | AI 예측":
    # Sidebar Navigation
    with st.sidebar:
        selected = option_menu(
            menu_title="Predictive Repetition | 예측 반복",
            options=["Data | 데이터", "Analyze | 분석하다"],
            icons=["clipboard-data", "graph-up-arrow"],
            menu_icon="cast",
            default_index=0,
            orientation="vertical"
        )

    # --------------------------- PAGE: DATA ---------------------------
    if selected == "Data | 데이터":
        st.title("🏋 Exercise Data Input Page | \n운동 데이터 입력 페이지")

        exercise_type = st.text_input("Enter Exercise Type | 운동 유형 입력", key="exercise_type")

        # Number of days
        num_days = st.sidebar.number_input("Enter Number of Days | 일수 입력", min_value=1, max_value=30, value=7, step=1)

        # Reps input
        st.subheader(f"Enter the number of reps for the next {num_days} days | \n다음 {num_days}일 동안의 반복 횟수 입력")

        input_reps = []
        for i in range(num_days):
            reps = st.number_input(f"Day {i + 1} Reps | 하루 {i + 1} 반복", min_value=0, step=1, key=f"day_{i + 1}")
            input_reps.append(reps)

        # Save button
        if st.button("💾 Save Data | 데이터 저장"):
            df = pd.DataFrame({
                "Day | 하루": list(range(1, num_days + 1)),
                "Reps | 반복": input_reps
            })
            st.session_state['exercise_data'] = df
            st.success("✅ Save Data! To Biew Statistics, go to the 'Analyze' Page. | 데이터 저장!통계를 보려면 '분석' 페이지로 이동합니다.")

    # ------------------------ PAGE: ANALYSIS --------------------------
    elif selected == "Analyze | 분석하다":
        st.title("📊 Exercise Analysis and Prediction | 운동 분석 및 예측")

        exercise_data = st.session_state.get('exercise_data', None)
        exercise_type = st.session_state.get('exercise_type', "Exercise")

        if exercise_data is not None and not exercise_data.empty:
            original_data = exercise_data.copy()
            original_data.index = original_data.index + 1
            original_data['Day'] = [f"Day {i}" for i in original_data.index]

            st.subheader("📋 User-Entered Reps Data | 사용자 입력 반복 데이터")
            st.dataframe(original_data.set_index('Day').transpose())

            # Bar Chart
            st.subheader("📈 Reps Chart | 반복 차트")
            colors = ['#0000FF', '#FF0000']
            cmap = LinearSegmentedColormap.from_list('my_gradient', colors, N=len(original_data))

            plt.figure(figsize=(10, 5))
            plt.bar(original_data['Day | 하루'], original_data['Reps | 반복'], color=cmap(np.linspace(0, 1, len(original_data))))
            plt.title("Exercise Reps")
            plt.xlabel("Days")
            plt.ylabel("Reps")
            plt.xticks(rotation=45)
            plt.grid(axis='y')
            st.pyplot(plt)

            # Statistics
            avg_reps = original_data['Reps | 반복'].mean()
            max_reps = original_data['Reps | 반복'].max()
            min_reps = original_data['Reps | 반복'].min()
            col1, col2, col3 = st.columns(3)
            col1.metric("Average Reps | 평균 반복", f"{avg_reps:.0f}")
            col2.metric("Max Reps | 최대 반복", f"{max_reps:.0f}")
            col3.metric("Min Reps | 최소 반복", f"{min_reps:.0f}")

            # ML Model selection
            model_type = st.sidebar.selectbox(
                "Select Prediction Model | 예측 모델 선택",
                ["Select... | 선택한다...", "Linear Regression", "Artificial Neural Network (ANN)", 
                "Support Vector Regression (SVR)", "K-Nearest Neighbors (KNN)", "Random Forest Regression"]
            )
            forecast_period = st.sidebar.number_input("Prediction Period (Days) | 예측 기간(일)", min_value=1, max_value=365, value=7)

            if model_type != "Select... | 선택한다..." and forecast_period > 0:
                y = original_data['Reps | 반복'].values.reshape(-1, 1)
                X = np.arange(len(y)).reshape(-1, 1)

                scaler = StandardScaler()
                y_scaled = scaler.fit_transform(y)

                X_train, X_test, y_train, y_test = train_test_split(X, y_scaled, test_size=0.3, random_state=42)

                if model_type == "Artificial Neural Network (ANN)":
                    model = Sequential([
                        Dense(20, activation='relu', input_shape=(X_train.shape[1],)),
                        Dropout(0.2),
                        Dense(10, activation='relu'),
                        Dropout(0.2),
                        Dense(5, activation='relu'),
                        Dense(1)
                    ])
                    model.compile(optimizer='adam', loss='mean_squared_error')
                    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
                    model.fit(X_train, y_train, epochs=100, batch_size=5, verbose=0,
                            validation_data=(X_test, y_test), callbacks=[early_stop])
                    future_days = np.arange(len(y), len(y) + forecast_period).reshape(-1, 1)
                    forecast_scaled = model.predict(future_days)

                elif model_type == "Linear Regression":
                    model = LinearRegression()
                    model.fit(X_train, y_train)
                    future_days = np.arange(len(y), len(y) + forecast_period).reshape(-1, 1)
                    forecast_scaled = model.predict(future_days)

                elif model_type == "Support Vector Regression (SVR)":
                    model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.01)
                    model.fit(X_train, y_train.ravel())
                    future_days = np.arange(len(y), len(y) + forecast_period).reshape(-1, 1)
                    forecast_scaled = model.predict(future_days).reshape(-1, 1)

                elif model_type == "K-Nearest Neighbors (KNN)":
                    model = KNeighborsRegressor(n_neighbors=5, weights='distance')
                    model.fit(X_train, y_train.ravel())
                    future_days = np.arange(len(y), len(y) + forecast_period).reshape(-1, 1)
                    forecast_scaled = model.predict(future_days).reshape(-1, 1)

                elif model_type == "Random Forest Regression":
                    model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
                    model.fit(X_train, y_train.ravel())
                    future_days = np.arange(len(y), len(y) + forecast_period).reshape(-1, 1)
                    forecast_scaled = model.predict(future_days).reshape(-1, 1)

                forecast = scaler.inverse_transform(forecast_scaled).flatten()
                forecast_df = pd.DataFrame({
                    'Day | 하루': [f"Day {i + 1}" for i in range(len(original_data), len(original_data) + forecast_period)],
                    'Reps | 반복': forecast.round(0)
                })

                st.subheader(f"📅 Forecasted Reps for Next {forecast_period} Days | \n다음 {forecast_period}일 동안의 예측된 담당자 수")
                st.dataframe(forecast_df.set_index('Day | 하루').transpose())

                # Forecast Chart
                plt.figure(figsize=(10, 5))
                plt.plot(forecast_df['Day | 하루'], forecast_df['Reps | 반복'], marker='o', color='b', label='Forecast')
                plt.title(f"{exercise_type} Forecast ({model_type})")
                plt.xlabel("Days")
                plt.ylabel("Predicted Reps")
                plt.xticks(rotation=45)
                plt.grid(True)
                plt.legend()
                st.pyplot(plt)

                # Stats
                col4, col5, col6 = st.columns(3)
                col4.metric("Average Reps | 평균 반복", f"{forecast.mean():.0f}")
                col5.metric("Max Reps | 최대 반복", f"{forecast.max():.0f}")
                col6.metric("Min Reps | 최소 반복", f"{forecast.min():.0f}")

                # Save as CSV
                if st.download_button(
                    label="📁 Download Predictions to CSV | CSV에 예측 다운로드",
                    data=forecast_df.to_csv(index=False).encode("utf-8"),
                    file_name=f"{exercise_type}_forecast.csv",
                    mime='text/csv'
                ):
                    st.success("Forecast CSV downloaded successfully | CSV 예측이 성공적으로 다운로드되었습니다")

            else:
                st.warning("⚠ Please select a model and forecast period to generate predictions | 예측을 생성할 모델과 예측 기간을 선택해 주세요")
        else:
            st.warning("⚠ No exercise data found. Go to the 'Data' page to enter reps | 운동 데이터를 찾을 수 없습니다. '데이터' 페이지로 이동하여 반복 입력합니다")
