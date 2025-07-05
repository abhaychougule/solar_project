from flask import Flask, render_template, Response, request, jsonify
import cv2
from datetime import datetime
import mysql.connector

# Import detection and training functions
from detection import detect_and_classify
from panel_training_code import train_panel_model
from panel_fault_detection import train_fault_model

app = Flask(__name__)

# USB camera
camera = cv2.VideoCapture(1)

# MySQL configuration
db_config = {
    'user': 'root',
    'password': 'Abhay123',
    'host': '127.0.0.1',
    'database': 'ranking_db'
}

def get_db_connection():
    return mysql.connector.connect(**db_config)

# Home Page
@app.route('/')
def index():
    return render_template('index.html')

# Video Feed Endpoint
@app.route('/video_feed')
def video_feed():
    def gen_frames():
        while True:
            success, frame = camera.read()
            if not success:
                break
            processed_frame, _ = detect_and_classify(frame)
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Fault Detection Endpoint
@app.route('/detect', methods=['POST'])
def detect():
    success, frame = camera.read()
    if not success:
        return jsonify({'error': 'Could not read from camera'})

    _, sections = detect_and_classify(frame)

    conn = get_db_connection()
    cursor = conn.cursor()

    fault_count = 0
    not_fault_count = 0
    total_loss = 0
    total_gain = 0
    detected_faults = []

    for section in sections:
        fault_name = section['fault_name']
        fault_value = section['fault_value']
        fault_section = section['fault_section']

        if fault_name != 'Clean':
            fault_count += 1
            total_loss += fault_value
        else:
            not_fault_count += 1
            total_gain += 20  # Example assumption

        cursor.execute("""
            INSERT INTO fault_tb (
                fault_name, fault_value, fault_section, date, 
                fault_count, not_fault_count, total_loss, total_gain
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """, (fault_name, fault_value, fault_section, datetime.now(),
              fault_count, not_fault_count, total_loss, total_gain))

        detected_faults.append({'fault_name': fault_name})

    conn.commit()
    cursor.close()
    conn.close()

    return jsonify({
        'status': 'success',
        'sections': detected_faults
    })

# Refresh Page
@app.route('/exit', methods=['POST'])
def exit_app():
    camera.release()
    cv2.destroyAllWindows()
    return "Camera released and window closed", 200

# Get Fault Data for Table
@app.route('/get_fault_data')
def get_fault_data():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT fault_name, fault_value, fault_section, date, 
               fault_count, not_fault_count, total_loss, total_gain 
        FROM fault_tb ORDER BY id DESC LIMIT 20
    """)
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return jsonify(rows)

# Panel Trainer Page
@app.route('/panel_trainer', methods=['GET', 'POST'])
def panel_trainer():
    if request.method == 'POST':
        train_path = request.form['train_path']
        test_path = request.form['test_path']
        train_panel_model(train_path, test_path)
        return "✅ Panel training completed successfully!"
    return render_template('panel_trainer.html')

# Fault Trainer Page
@app.route('/fault_trainer', methods=['GET', 'POST'])
def fault_trainer():
    if request.method == 'POST':
        train_path = request.form['train_path']
        test_path = request.form['test_path']
        train_fault_model(train_path, test_path)
        return "✅ Fault classification training completed successfully!"
    return render_template('fault_trainer.html')

# About Project Page
@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)
