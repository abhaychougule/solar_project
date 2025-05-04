from flask import Flask, render_template, Response, request, jsonify
from detection import detect_and_classify
import cv2
import mysql.connector
from datetime import datetime

app = Flask(__name__)

camera = cv2.VideoCapture(0)  # Use 0 for USB Camera

# MySQL Database configuration
db_config = {
    'user': 'root',
    'password': 'Abhay123',
    'host': '127.0.0.1',
    'database': 'ranking_db'
}

def get_db_connection():
    conn = mysql.connector.connect(**db_config)
    return conn

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    def gen_frames():
        while True:
            success, frame = camera.read()
            if not success:
                break
            else:
                # Perform detection and classification
                processed_frame, sections = detect_and_classify(frame)

                ret, buffer = cv2.imencode('.jpg', processed_frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detect', methods=['POST'])
def detect():
    # Take the latest frame
    success, frame = camera.read()
    if not success:
        return jsonify({'error': 'Could not read from camera'})

    # Perform detection and classification
    _, sections = detect_and_classify(frame)

    # Process and store results
    conn = get_db_connection()
    cursor = conn.cursor()

    fault_count = 0
    not_fault_count = 0
    total_loss = 0
    total_gain = 0
    
    # List to hold the detected faults for the response
    detected_faults = []
    
    for section in sections:
        fault_name = section['fault_name']
        fault_value = section['fault_value']
        fault_section = section['fault_section']
        
        # Count faults
        if fault_name != 'Clean':
            fault_count += 1
            total_loss += fault_value
        else:
            not_fault_count += 1
            total_gain += 20  # Assuming each clean section generates 20 MW
        
        # Insert into database
        cursor.execute("""
            INSERT INTO fault_tb (fault_name, fault_value, fault_section, date, fault_count, not_fault_count, total_loss, total_gain)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """, (fault_name, fault_value, fault_section, datetime.now(), fault_count, not_fault_count, total_loss, total_gain))
        
        detected_faults.append({'fault_name': fault_name})
    
    conn.commit()
    cursor.close()
    conn.close()

    # return jsonify({'status': 'success'})
    return jsonify({
        'status': 'success',
        'sections': detected_faults  # Send detected fault names for each section
    })

@app.route('/exit', methods=['POST'])
def exit():
    camera.release()
    cv2.destroyAllWindows()
    return "Camera released and window closed", 200

@app.route('/get_fault_data', methods=['GET'])
def get_fault_data():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT fault_name, fault_value, fault_section, date, fault_count, not_fault_count, total_loss, total_gain FROM fault_tb ORDER BY id DESC LIMIT 20")
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return jsonify(rows)

if __name__ == '__main__':
    app.run(debug=True)
