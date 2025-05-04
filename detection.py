import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load models
panel_model = load_model('Classifiers/solar_panel_detection_model.h5')
fault_model = load_model('Classifiers/solar_fault_detection_model.h5')

# Load labels
with open('Classifiers/labels.txt') as f:
    panel_labels = [line.strip() for line in f]

with open('Classifiers/fault_labels.txt') as f:
    fault_labels = [line.strip() for line in f]

# Fault value mapping
fault_values = {
    "Bird-drop": 20,
    "Clean": 0,
    "Dusty": 20,
    "Electrical-damage": 20,
    "Physical-Damage": 20,
    "Snow-Covered": 20
}

def detect_and_classify(frame):
    height, width, _ = frame.shape
    
    # Preprocess the frame for panel detection
    resized_frame = cv2.resize(frame, (150, 150))
    panel_input = np.expand_dims(resized_frame / 255.0, axis=0)
    
    # Predict panel presence
    panel_pred = panel_model.predict(panel_input)
    panel_class = panel_labels[np.argmax(panel_pred)]
    
    sections = []

    if panel_class == "Solar_Panel":
        cv2.putText(frame, "Solar Panel Detected", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Create a green bounding box around the panel
        cv2.rectangle(frame, (10, 10), (width-10, height-10), (0, 255, 0), 2)
        
        # Divide the panel into 9 sections and detect faults
        section_height = height // 3
        section_width = width // 3
        
        for i in range(3):
            for j in range(3):
                x1, y1 = j * section_width, i * section_height
                x2, y2 = x1 + section_width, y1 + section_height
                section = frame[y1:y2, x1:x2]
                
                # Preprocess section for fault detection
                resized_section = cv2.resize(section, (150, 150))
                section_input = np.expand_dims(resized_section / 255.0, axis=0)
                
                # Predict fault in the section
                fault_pred = fault_model.predict(section_input)
                fault_class = fault_labels[np.argmax(fault_pred)]
                
                # Draw red bounding box and label for each section
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, fault_class, (x1 + 10, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                # Store section details
                section_id = f"Section_{i}_{j}"
                sections.append({
                    'fault_name': fault_class,
                    'fault_value': fault_values[fault_class],
                    'fault_section': section_id
                })
    
    return frame, sections
