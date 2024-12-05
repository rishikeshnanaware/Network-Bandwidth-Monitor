import matplotlib.pyplot as plt
from collections import deque
import psutil
import time
import csv
from datetime import datetime
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# Function to get current network statistics
def get_network_stats():
    stats = psutil.net_io_counters()
    return stats.bytes_sent, stats.bytes_recv

# Function to log network statistics into a CSV file
def log_network_stats(filename, timestamp, upload_speed, download_speed, total_upload, total_download):
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([timestamp, upload_speed, download_speed, total_upload, total_download])

# Function to predict future network usage based on past data
def predict_network_usage(upload_speeds, download_speeds, future_steps=10):
    if len(upload_speeds) < 2 or len(download_speeds) < 2:
        print("Not enough data for prediction.")
        return [], []

    # Prepare data
    x = np.arange(len(upload_speeds)).reshape(-1, 1)
    y_upload = np.array(upload_speeds)
    y_download = np.array(download_speeds)

    # Train models
    model_upload = LinearRegression().fit(x, y_upload)
    model_download = LinearRegression().fit(x, y_download)

    # Predict future values
    future_x = np.arange(len(upload_speeds), len(upload_speeds) + future_steps).reshape(-1, 1)
    future_upload = model_upload.predict(future_x)
    future_download = model_download.predict(future_x)

    return future_upload, future_download

# Main function to monitor network usage and generate predictions
def monitor_network_usage_with_prediction(interval=1, log_file='network_usage.csv', duration=60):
    # Initialize CSV file with headers
    try:
        with open(log_file, mode='x', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Timestamp", "Upload Speed (Bytes/s)", "Download Speed (Bytes/s)", "Total Upload (Bytes)", "Total Download (Bytes)"])
    except FileExistsError:
        pass

    prev_sent, prev_recv = get_network_stats()
    total_sent, total_recv = prev_sent, prev_recv

    # Initialize plotting variables
    timestamps = deque(maxlen=60)
    upload_speeds = deque(maxlen=60)
    download_speeds = deque(maxlen=60)

    plt.ion()
    fig, ax = plt.subplots()
    line1, = ax.plot([], [], label="Upload Speed (B/s)", color="blue")
    line2, = ax.plot([], [], label="Download Speed (B/s)", color="orange")
    ax.legend()
    ax.set_xlabel("Time")
    ax.set_ylabel("Speed (Bytes/s)")

    print("Monitoring network usage...")

    for _ in range(int(duration / interval)):
        time.sleep(interval)

        curr_sent, curr_recv = get_network_stats()

        upload_speed = (curr_sent - prev_sent) / interval
        download_speed = (curr_recv - prev_recv) / interval

        total_sent = curr_sent
        total_recv = curr_recv

        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Log network stats to CSV
        log_network_stats(log_file, timestamp, upload_speed, download_speed, total_sent, total_recv)

        # Print current stats
        print(f"{timestamp} | Upload Speed: {upload_speed:.2f} B/s | Download Speed: {download_speed:.2f} B/s | Total Upload: {total_sent / (1024 ** 2):.2f} MB | Total Download: {total_recv / (1024 ** 2):.2f} MB")

        # Update plotting data
        timestamps.append(timestamp)
        upload_speeds.append(upload_speed)
        download_speeds.append(download_speed)

        # Update plot
        line1.set_data(range(len(upload_speeds)), upload_speeds)
        line2.set_data(range(len(download_speeds)), download_speeds)
        ax.relim()
        ax.autoscale_view()
        plt.draw()
        plt.pause(0.01)

        # Predict future usage after sufficient data is collected
        if len(upload_speeds) >= 10:
            future_upload, future_download = predict_network_usage(upload_speeds, download_speeds, future_steps=5)
            print("Predicted Future Upload Speeds:", future_upload)
            print("Predicted Future Download Speeds:", future_download)

        prev_sent, prev_recv = curr_sent, curr_recv

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    monitor_network_usage_with_prediction(interval=1, log_file='network_usage.csv', duration=60)
