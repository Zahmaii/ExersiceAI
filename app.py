import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from streamlit_webrtc import webrtc_streamer
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
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from matplotlib.colors import LinearSegmentedColormap
from streamlit_option_menu import option_menu

selected = option_menu(
    menu_title="Exercise AI Tools | Exercise AI Tools",  
    options=["Analyzer", "Detection", "Prediction"],  
    icons=["clipboard-data", "graph-up-arrow", "calculator-fill"],  
    menu_icon="cast",
    default_index=0,  
    orientation="horizontol"
)

if selected == "Analyzer":
    st.title("📏 Posture Analyzer for Squats & Sit-ups")

    # MediaPipe setup
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    
    # Helper function to calculate joint angles
    def calculate_angle(a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180.0:
            angle = 360 - angle
        return angle

    # Helper function to compute posture score
    def get_posture_score(exercise, angle):
        if exercise == "Squat":
            # Target knee angle: ~90° when down, ~160° when standing
            if angle < 80 or angle > 170:
                return angle, 50, "Try to keep your knees at a 90° angle at the bottom of the squat."
            elif 80 <= angle <= 100 or 160 <= angle <= 180:
                return angle, 90, "Great form! Keep it up."
            else:
                return angle, 70, "Adjust your depth for a better range of motion."
        elif exercise == "Sit-up":
            # Hip angle: closer to 45° (bent) and 180° (flat)
            if angle > 150:
                return angle, 90, "Good sit-up posture. Full extension."
            elif angle < 100:
                return angle, 50, "You may be curling too much. Try to control your core."
            else:
                return angle, 70, "Decent posture. Aim for smoother range."
        return angle, 0, "No exercise selected."

    # Sidebar
    st.sidebar.title("Options")
    exercise_type = st.sidebar.selectbox("Select Exercise", ["Select...", "Squat", "Sit-up"])
    mode = st.sidebar.radio("Choose Mode", ["Live Camera", "Upload Video"])
    record_session = st.sidebar.checkbox("Record Session")

    FRAME_WINDOW = st.image([])

    # ----------------- Live Camera Mode -------------------
    if exercise_type != "Select..." and mode == "Live Camera":
        st.info(f"Perform **{exercise_type}s** in front of your webcam. Click below to start.")

        start_cam = st.button("Start Camera")
        if start_cam:
            cap = cv2.VideoCapture(0)

            if record_session:
                temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".avi")
                out = cv2.VideoWriter(
                    temp_video.name,
                    cv2.VideoWriter_fourcc(*'XVID'),
                    20.0,
                    (640, 480)
                )
            else:
                out = None

            with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image.flags.writeable = False
                    results = pose.process(image)
                    image.flags.writeable = True
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                    try:
                        landmarks = results.pose_landmarks.landmark

                        if exercise_type == "Squat":
                            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                            knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                                    landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                            ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                                    landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                            angle = calculate_angle(hip, knee, ankle)

                        elif exercise_type == "Sit-up":
                            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                            knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                                    landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                            angle = calculate_angle(shoulder, hip, knee)

                        angle, score, feedback = get_posture_score(exercise_type, angle)

                        cv2.putText(image, f"Angle: {int(angle)}", (30, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                        cv2.putText(image, f"Score: {score}", (30, 70),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 255, 100), 2)
                        cv2.putText(image, f"Feedback: {feedback}", (30, 100),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 255), 2)

                        mp_drawing.draw_landmarks(
                            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                            mp_drawing.DrawingSpec(color=(66, 245, 230), thickness=2, circle_radius=2)
                        )

                    except:
                        pass

                    if out:
                        out.write(image)

                    FRAME_WINDOW.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

                cap.release()
                if out:
                    out.release()
                cv2.destroyAllWindows()

            st.success("Camera session ended.")
            if record_session:
                st.video(temp_video.name)

    # ----------------- Upload Video Mode -------------------
    elif exercise_type != "Select..." and mode == "Upload Video":
        st.info("Upload a recorded workout video for posture analysis.")
        uploaded_video = st.file_uploader("Upload your exercise video (.mp4, .avi)", type=["mp4", "avi"])

        if uploaded_video:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_video.read())

            cap = cv2.VideoCapture(tfile.name)
            FRAME_WINDOW = st.image([])

            with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image.flags.writeable = False
                    results = pose.process(image)
                    image.flags.writeable = True
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                    try:
                        landmarks = results.pose_landmarks.landmark

                        if exercise_type == "Squat":
                            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                            knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                                    landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                            ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                                    landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                            angle = calculate_angle(hip, knee, ankle)

                        elif exercise_type == "Sit-up":
                            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                            knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                                    landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                            angle = calculate_angle(shoulder, hip, knee)

                        angle, score, feedback = get_posture_score(exercise_type, angle)

                        cv2.putText(image, f"Angle: {int(angle)}", (30, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                        cv2.putText(image, f"Score: {score}", (30, 70),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 255, 100), 2)
                        cv2.putText(image, f"Feedback: {feedback}", (30, 100),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 255), 2)

                        mp_drawing.draw_landmarks(
                            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                            mp_drawing.DrawingSpec(color=(66, 245, 230), thickness=2, circle_radius=2)
                        )

                    except:
                        pass

                    FRAME_WINDOW.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

                cap.release()
                cv2.destroyAllWindows()

            st.success("Video analysis complete.")

elif selected == "Detection":

    # Constants
    CSV_FILE = "exercise_history.csv"

    # Initialize CSV if not exist
    if "initialized" not in st.session_state:
        try:
            pd.read_csv(CSV_FILE)
        except FileNotFoundError:
            df_init = pd.DataFrame(columns=["Date", "Exercise", "Reps"])
            df_init.to_csv(CSV_FILE, index=False)
        st.session_state.initialized = True

    st.title("🏋️ Exercise Detector & Rep Counter")

    # Sidebar
    exercise_type = st.sidebar.selectbox(
        "Select Exercise",
        ["Select...", "Push-up", "Squat"]
    )

    enable_timer = st.sidebar.checkbox("⏲ Enable Timer")
    if enable_timer:
        duration = st.sidebar.number_input("Set Timer (seconds)", min_value=10, value=30)

    stop_session = st.sidebar.button("⛔ Stop Session")

    # Session state
    if "counter" not in st.session_state:
        st.session_state.counter = 0
    if "stage" not in st.session_state:
        st.session_state.stage = None
    if "start_time" not in st.session_state:
        st.session_state.start_time = None
    if "run_session" not in st.session_state:
        st.session_state.run_session = False

    def calculate_angle(a, b, c):
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)

        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)

        if angle > 180.0:
            angle = 360 - angle

        return angle

    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    if exercise_type != "Select...":
        st.info(f"Start performing **{exercise_type}s** in front of your webcam.")
        
        if not st.session_state.run_session:
            start_button = st.button("▶️ Start Session")
            if start_button:
                st.session_state.run_session = True
                st.session_state.counter = 0
                st.session_state.stage = None
                st.session_state.start_time = time.time()
        else:
            if stop_session:
                st.session_state.run_session = False

        def video_frame_callback(frame):
            img = frame.to_ndarray(format="bgr24")

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
                results = pose.process(img_rgb)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                try:
                    if exercise_type == "Squat":
                        hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                               landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                        knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                        ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                                 landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                        angle = calculate_angle(hip, knee, ankle)
                        cv2.putText(img, str(int(angle)),
                                    tuple(np.multiply(knee, [img.shape[1], img.shape[0]]).astype(int)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                        if angle > 160:
                            st.session_state.stage = "up"
                        if angle < 120 and st.session_state.stage == 'up':
                            st.session_state.stage = "down"
                            st.session_state.counter += 1

                    elif exercise_type == "Push-up":
                        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                        elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                                 landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                        wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                                 landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                        angle = calculate_angle(shoulder, elbow, wrist)
                        cv2.putText(img, str(int(angle)),
                                    tuple(np.multiply(elbow, [img.shape[1], img.shape[0]]).astype(int)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                        if angle > 160:
                            st.session_state.stage = "up"
                        if angle < 90 and st.session_state.stage == 'up':
                            st.session_state.stage = "down"
                            st.session_state.counter += 1

                    mp_drawing.draw_landmarks(
                        img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2),
                        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2)
                    )
                except:
                    pass

            # Timer
            if enable_timer and st.session_state.start_time is not None:
                elapsed = time.time() - st.session_state.start_time
                remaining = int(duration - elapsed)
                cv2.putText(img, f"Time left: {remaining}s", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                if remaining <= 0:
                    st.session_state.run_session = False

            # Counter box
            cv2.rectangle(img, (0, 0), (250, 80), (245, 117, 16), -1)
            cv2.putText(img, 'REPS', (15, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
            cv2.putText(img, str(st.session_state.counter), (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)

            return img

        if st.session_state.run_session:
            webrtc_streamer(
                key="exercise-detection",
                video_frame_callback=video_frame_callback,
                media_stream_constraints={"video": True, "audio": False},
            )
        else:
            st.write("Session stopped or not started.")

        # Save to CSV and show history when session ends
        if not st.session_state.run_session and st.session_state.counter > 0:
            st.success(f"Session ended. Total reps counted: {st.session_state.counter}")
            df_new = pd.DataFrame({
                "Date": [datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
                "Exercise": [exercise_type],
                "Reps": [st.session_state.counter]
            })
            df_existing = pd.read_csv(CSV_FILE)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            df_combined.to_csv(CSV_FILE, index=False)

            st.write("✅ Session saved to history.")

            st.subheader(f"📈 Past History for {exercise_type}")
            filtered = df_combined[df_combined["Exercise"] == exercise_type]
            st.dataframe(filtered.tail(10), use_container_width=True)
    else:
        st.warning("👈 Please select an exercise to begin.")

        
elif selected == "Prediction":
    # Sidebar Navigation
    with st.sidebar:
        selected = option_menu(
            menu_title="Reps Prediction",
            options=["Data", "Analysis"],
            icons=["clipboard-data", "graph-up-arrow"],
            menu_icon="cast",
            default_index=0,
            orientation="vertical"
        )

    # --------------------------- PAGE: DATA ---------------------------
    if selected == "Data":
        st.title("🏋️ Exercise Data Input Page")

        exercise_type = st.text_input("Enter the Exercise Type", key="exercise_type")

        # Number of days
        num_days = st.sidebar.number_input("Enter Number of Days", min_value=1, max_value=30, value=7, step=1)

        # Reps input
        st.subheader(f"Enter the number of reps for {exercise_type or 'the exercise'} over {num_days} days")

        input_reps = []
        for i in range(num_days):
            reps = st.number_input(f"Day {i + 1} Reps", min_value=0, step=1, key=f"day_{i + 1}")
            input_reps.append(reps)

        # Save button
        if st.button("💾 Save Data"):
            df = pd.DataFrame({
                "Day": list(range(1, num_days + 1)),
                "Reps": input_reps
            })
            st.session_state['exercise_data'] = df
            st.success("✅ Data saved! Go to the 'Analysis' page to view your stats.")

    # ------------------------ PAGE: ANALYSIS --------------------------
    elif selected == "Analysis":
        st.title("📊 Exercise Analysis and Forecasting")

        exercise_data = st.session_state.get('exercise_data', None)
        exercise_type = st.session_state.get('exercise_type', "Exercise")

        if exercise_data is not None and not exercise_data.empty:
            original_data = exercise_data.copy()
            original_data.index = original_data.index + 1
            original_data['Day'] = [f"Day {i}" for i in original_data.index]

            st.subheader("📋 User-Entered Reps Data")
            st.dataframe(original_data.set_index('Day').transpose())

            # Bar Chart
            st.subheader("📈 Reps Chart")
            colors = ['#0000FF', '#FF0000']
            cmap = LinearSegmentedColormap.from_list('my_gradient', colors, N=len(original_data))

            plt.figure(figsize=(10, 5))
            plt.bar(original_data['Day'], original_data['Reps'], color=cmap(np.linspace(0, 1, len(original_data))))
            plt.title("Exercise Reps")
            plt.xlabel("Days")
            plt.ylabel("Reps")
            plt.xticks(rotation=45)
            plt.grid(axis='y')
            st.pyplot(plt)

            # Statistics
            avg_reps = original_data['Reps'].mean()
            max_reps = original_data['Reps'].max()
            min_reps = original_data['Reps'].min()
            col1, col2, col3 = st.columns(3)
            col1.metric("Average Reps", f"{avg_reps:.0f}")
            col2.metric("Max Reps", f"{max_reps:.0f}")
            col3.metric("Min Reps", f"{min_reps:.0f}")

            # ML Model selection
            model_type = st.sidebar.selectbox(
                "Select Prediction Model",
                ["Select...", "Linear Regression", "Artificial Neural Network (ANN)", "Support Vector Regression (SVR)"]
            )
            forecast_period = st.sidebar.number_input("Forecast Period (Days)", min_value=1, max_value=365, value=7)

            if model_type != "Select..." and forecast_period > 0:
                y = original_data['Reps'].values.reshape(-1, 1)
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

                forecast = scaler.inverse_transform(forecast_scaled).flatten()
                forecast_df = pd.DataFrame({
                    'Day': [f"Day {i + 1}" for i in range(len(original_data), len(original_data) + forecast_period)],
                    'Reps': forecast.round(0)
                })

                st.subheader(f"📅 Forecasted Reps for Next {forecast_period} Days")
                st.dataframe(forecast_df.set_index('Day').transpose())

                # Forecast Chart
                plt.figure(figsize=(10, 5))
                plt.plot(forecast_df['Day'], forecast_df['Reps'], marker='o', color='b', label='Forecast')
                plt.title(f"{exercise_type} Forecast ({model_type})")
                plt.xlabel("Days")
                plt.ylabel("Predicted Reps")
                plt.xticks(rotation=45)
                plt.grid(True)
                plt.legend()
                st.pyplot(plt)

                # Stats
                col4, col5, col6 = st.columns(3)
                col4.metric("Average Reps", f"{forecast.mean():.0f}")
                col5.metric("Max Reps", f"{forecast.max():.0f}")
                col6.metric("Min Reps", f"{forecast.min():.0f}")

                # Save as CSV
                if st.download_button(
                    label="📁 Download Forecast as CSV",
                    data=forecast_df.to_csv(index=False).encode("utf-8"),
                    file_name=f"{exercise_type}_forecast.csv",
                    mime='text/csv'
                ):
                    st.success("Forecast CSV downloaded successfully.")

            else:
                st.warning("⚠️ Please select a model and forecast period to generate predictions.")
        else:
            st.warning("⚠️ No exercise data found. Go to the 'Data' page to enter reps.")
