import tkinter as tk
from tkinter import messagebox
from datetime import datetime, timedelta
import time
import threading
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import os
import simpleaudio as sa
import requests
import random
import speech_recognition as sr
from sklearn.cluster import KMeans
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

class AlarmClockAI:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Alarm Clock")
        self.root.geometry("700x700")

        self.time_label = tk.Label(self.root, text="Set Alarm Time (24-hour format):", font=("Helvetica", 14))
        self.time_label.pack(pady=10)

        self.time_entry = tk.Entry(self.root, font=("Helvetica", 14))
        self.time_entry.pack(pady=5)

        self.message_label = tk.Label(self.root, text="Custom Message:", font=("Helvetica", 14))
        self.message_label.pack(pady=10)

        self.message_entry = tk.Entry(self.root, font=("Helvetica", 14))
        self.message_entry.pack(pady=5)

        self.days_label = tk.Label(self.root, text="Repeat on (comma-separated weekdays, e.g., Mon,Wed,Fri):", font=("Helvetica", 14))
        self.days_label.pack(pady=10)

        self.days_entry = tk.Entry(self.root, font=("Helvetica", 14))
        self.days_entry.pack(pady=5)

        self.set_button = tk.Button(self.root, text="Set Alarm", command=self.set_alarm, font=("Helvetica", 14))
        self.set_button.pack(pady=10)

        self.train_button = tk.Button(self.root, text="Train AI", command=self.train_ai, font=("Helvetica", 14))
        self.train_button.pack(pady=10)

        self.predict_button = tk.Button(self.root, text="Predict Wake-Up Time", command=self.predict_wake_up_time, font=("Helvetica", 14))
        self.predict_button.pack(pady=10)

        self.smart_schedule_button = tk.Button(self.root, text="Smart Schedule Alarm", command=self.smart_schedule_alarm, font=("Helvetica", 14))
        self.smart_schedule_button.pack(pady=10)

        self.analyze_sleep_button = tk.Button(self.root, text="Analyze Sleep Pattern", command=self.analyze_sleep_pattern, font=("Helvetica", 14))
        self.analyze_sleep_button.pack(pady=10)

        self.weather_adjust_button = tk.Button(self.root, text="Adjust Alarm for Weather", command=self.adjust_alarm_for_weather, font=("Helvetica", 14))
        self.weather_adjust_button.pack(pady=10)

        self.mood_predict_button = tk.Button(self.root, text="Predict Mood", command=self.predict_mood, font=("Helvetica", 14))
        self.mood_predict_button.pack(pady=10)

        self.smart_bedtime_button = tk.Button(self.root, text="Suggest Bedtime", command=self.suggest_bedtime, font=("Helvetica", 14))
        self.smart_bedtime_button.pack(pady=10)

        self.stress_predict_button = tk.Button(self.root, text="Predict Stress Level", command=self.predict_stress_level, font=("Helvetica", 14))
        self.stress_predict_button.pack(pady=10)

        self.exercise_recommend_button = tk.Button(self.root, text="Recommend Exercise", command=self.recommend_exercise, font=("Helvetica", 14))
        self.exercise_recommend_button.pack(pady=10)

        self.quote_button = tk.Button(self.root, text="Generate Motivational Quote", command=self.generate_quote, font=("Helvetica", 14))
        self.quote_button.pack(pady=10)

        self.alarms_listbox = tk.Listbox(self.root, font=("Helvetica", 14))
        self.alarms_listbox.pack(pady=10)

        self.deactivate_button = tk.Button(self.root, text="Deactivate Alarm", command=self.deactivate_alarm, font=("Helvetica", 14))
        self.deactivate_button.pack(pady=10)

        self.stop_button = tk.Button(self.root, text="Stop Alarm", command=self.stop_alarm, font=("Helvetica", 14))
        self.stop_button.pack(pady=10)

        self.voice_command_button = tk.Button(self.root, text="Set Alarm via Voice", command=self.set_alarm_via_voice, font=("Helvetica", 14))
        self.voice_command_button.pack(pady=10)

        self.alarm_times = []
        self.custom_messages = []
        self.recurring_days = []
        self.alarm_set = False
        self.alarm_thread = None
        self.data_file = "alarm_data.csv"
        self.quotes = [
            "The best way to get started is to quit talking and begin doing. - Walt Disney",
            "The pessimist sees difficulty in every opportunity. The optimist sees opportunity in every difficulty. - Winston Churchill",
            "Don’t let yesterday take up too much of today. - Will Rogers"
        ]

        if not os.path.exists(self.data_file):
            self.init_data_file()

        self.df = pd.read_csv(self.data_file)
        self.model = LinearRegression()
        self.kmeans_model = None
        self.lstm_model = None
        self.mood_model = RandomForestClassifier()
        self.stress_model = RandomForestClassifier()
        self.exercise_model = RandomForestClassifier()

    def init_data_file(self):
        df = pd.DataFrame(columns=['day_of_week', 'hour', 'minute', 'mood', 'stress_level'])
        df.to_csv(self.data_file, index=False)

    def set_alarm(self):
        alarm_time_str = self.time_entry.get()
        custom_message = self.message_entry.get()
        days_str = self.days_entry.get()
        try:
            alarm_time = datetime.strptime(alarm_time_str, "%H:%M")
            self.alarm_times.append(alarm_time)
            self.custom_messages.append(custom_message)
            self.recurring_days.append(days_str.split(","))
            self.alarms_listbox.insert(tk.END, f"Alarm set for {alarm_time_str} - {custom_message} - {days_str}")
            messagebox.showinfo("Alarm Set", f"Alarm set for {alarm_time.strftime('%H:%M')} on {days_str}")
            self.save_alarm_time(alarm_time)
            self.check_alarm()
        except ValueError:
            messagebox.showerror("Invalid Time", "Please enter a valid time in HH:MM format.")

    def save_alarm_time(self, alarm_time):
        now = datetime.now()
        mood = random.choice(['happy', 'neutral', 'tired'])  # Mock mood data
        stress_level = random.choice(['low', 'medium', 'high'])  # Mock stress data
        alarm_data = {'day_of_week': [now.weekday()], 'hour': [alarm_time.hour], 'minute': [alarm_time.minute], 'mood': [mood], 'stress_level': [stress_level]}
        df_new = pd.DataFrame(alarm_data)
        self.df = pd.concat([self.df, df_new], ignore_index=True)
        self.df.to_csv(self.data_file, index=False)

    def train_ai(self):
        X = self.df[['day_of_week']]
        y = self.df[['hour', 'minute']]
        self.model.fit(X, y)
        
        kmeans_X = self.df[['hour', 'minute']]
        self.kmeans_model = KMeans(n_clusters=3)
        self.kmeans_model.fit(kmeans_X)

        # Train LSTM model for advanced prediction
        self.train_lstm_model()

        # Train mood prediction model
        self.train_mood_model()

        # Train stress prediction model
        self.train_stress_model()

        # Train exercise recommendation model
        self.train_exercise_model()
        
        messagebox.showinfo("AI Training", "AI model has been trained on your wake-up data.")

    def train_lstm_model(self):
        df = self.df.copy()
        df['time'] = df['hour'] * 60 + df['minute']
        data = df[['time']].values
        generator = TimeseriesGenerator(data, data, length=3, batch_size=1)
        
        model = Sequential([
            LSTM(50, activation='relu', input_shape=(3, 1)),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(generator, epochs=50)
        
        self.lstm_model = model

    def train_mood_model(self):
        features = self.df[['day_of_week', 'hour', 'minute']]
        target = self.df['mood']
        self.mood_model.fit(features, target)

    def train_stress_model(self):
        features = self.df[['day_of_week', 'hour', 'minute']]
        target = self.df['stress_level']
        self.stress_model.fit(features, target)

    def train_exercise_model(self):
        features = self.df[['day_of_week', 'hour', 'minute']]
        target = np.random.choice(['yoga', 'running', 'stretching'], size=len(features))  # Mock exercise data
        self.exercise_model.fit(features, target)

    def predict_wake_up_time(self):
        now = datetime.now()
        X_new = pd.DataFrame({'day_of_week': [now.weekday()]})
        y_pred = self.model.predict(X_new)
        predicted_time = f"{int(y_pred[0][0]):02}:{int(y_pred[0][1]):02}"
        messagebox.showinfo("Predicted Wake-Up Time", f"Predicted wake-up time: {predicted_time}")

    def smart_schedule_alarm(self):
        now = datetime.now()
        day_of_week = now.weekday()
        time_cluster = self.kmeans_model.predict([[now.hour, now.minute]])[0]
        cluster_center = self.kmeans_model.cluster_centers_[time_cluster]
        optimal_time = f"{int(cluster_center[0]):02}:{int(cluster_center[1]):02}"
        messagebox.showinfo("Smart Schedule", f"Optimal wake-up time: {optimal_time}")

    def analyze_sleep_pattern(self):
        sleep_pattern = self.df.groupby('day_of_week')['hour'].mean()
        pattern_message = "\n".join([f"Day {int(day)}: {int(hour)} hours" for day, hour in sleep_pattern.items()])
        messagebox.showinfo("Sleep Pattern Analysis", f"Average sleep pattern:\n{pattern_message}")

    def adjust_alarm_for_weather(self):
        api_key = "your_openweather_api_key"
        city = "your_city"
        response = requests.get(f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}")
        weather_data = response.json()
        weather_condition = weather_data['weather'][0]['description']
        now = datetime.now()
        alarm_time = now + timedelta(minutes=30)  # Adjust alarm time by 30 minutes for adverse weather
        messagebox.showinfo("Weather Adjustment", f"Weather condition: {weather_condition}. Alarm adjusted to {alarm_time.strftime('%H:%M')}.")

    def predict_mood(self):
        now = datetime.now()
        X_new = pd.DataFrame({'day_of_week': [now.weekday()], 'hour': [now.hour], 'minute': [now.minute]})
        mood_pred = self.mood_model.predict(X_new)[0]
        messagebox.showinfo("Mood Prediction", f"Predicted mood: {mood_pred}")

    def suggest_bedtime(self):
        now = datetime.now()
        wake_up_time = now + timedelta(hours=8)  # Assuming 8 hours of sleep is optimal
        messagebox.showinfo("Suggested Bedtime", f"Suggested bedtime: {wake_up_time.strftime('%H:%M')}")

    def predict_stress_level(self):
        now = datetime.now()
        X_new = pd.DataFrame({'day_of_week': [now.weekday()], 'hour': [now.hour], 'minute': [now.minute]})
        stress_pred = self.stress_model.predict(X_new)[0]
        messagebox.showinfo("Stress Prediction", f"Predicted stress level: {stress_pred}")

    def recommend_exercise(self):
        now = datetime.now()
        X_new = pd.DataFrame({'day_of_week': [now.weekday()], 'hour': [now.hour], 'minute': [now.minute]})
        exercise_pred = self.exercise_model.predict(X_new)[0]
        messagebox.showinfo("Exercise Recommendation", f"Recommended exercise: {exercise_pred}")

    def generate_quote(self):
        api_url = "https://api.quotable.io/random"
        response = requests.get(api_url)
        if response.status_code == 200:
            quote = response.json()['content']
            messagebox.showinfo("Motivational Quote", quote)
        else:
            messagebox.showinfo("Motivational Quote", random.choice(self.quotes))

    def check_alarm(self):
        if not self.alarm_set:
            self.alarm_set = True
            self.alarm_thread = threading.Thread(target=self.monitor_alarms)
            self.alarm_thread.start()

    def monitor_alarms(self):
        while self.alarm_set:
            now = datetime.now().time()
            for i, alarm_time in enumerate(self.alarm_times):
                if now.hour == alarm_time.hour and now.minute == alarm_time.minute:
                    days = self.recurring_days[i]
                    if datetime.now().strftime("%a") in days:
                        self.trigger_alarm(self.custom_messages[i])
            time.sleep(10)

    def trigger_alarm(self, message):
        wave_obj = sa.WaveObject.from_wave_file("alarm_sound.wav")
        play_obj = wave_obj.play()
        play_obj.wait_done()
        messagebox.showinfo("Alarm", f"Wake up! {message}")

    def deactivate_alarm(self):
        self.alarm_set = False
        if self.alarm_thread is not None:
            self.alarm_thread.join()
        self.alarm_times = []
        self.custom_messages = []
        self.recurring_days = []
        self.alarms_listbox.delete(0, tk.END)
        messagebox.showinfo("Deactivate Alarm", "All alarms have been deactivated.")

    def stop_alarm(self):
        sa.stop_all()
        messagebox.showinfo("Stop Alarm", "Alarm stopped.")

    def set_alarm_via_voice(self):
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            messagebox.showinfo("Voice Command", "Listening for alarm time...")
            audio = recognizer.listen(source)
        try:
            voice_input = recognizer.recognize_google(audio)
            alarm_time_str = voice_input.split(" ")[0]
            alarm_time = datetime.strptime(alarm_time_str, "%H:%M")
            self.alarm_times.append(alarm_time)
            self.custom_messages.append("Voice set alarm")
            self.alarms_listbox.insert(tk.END, f"Voice set alarm for {alarm_time_str}")
            messagebox.showinfo("Alarm Set", f"Alarm set for {alarm_time.strftime('%H:%M')} via voice command.")
            self.check_alarm()
        except Exception as e:
            messagebox.showerror("Voice Command Error", f"Could not understand voice command. Error: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    alarm_clock = AlarmClockAI(root)
    root.mainloop()
