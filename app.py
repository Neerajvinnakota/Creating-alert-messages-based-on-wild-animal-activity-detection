import cv2
import torch
import smtplib
from email.mime.text import MIMEText
from ultralytics import YOLO
from flask import Flask, render_template, Response

app = Flask(__name__)

# Load pre-trained YOLO model for animal detection
model = YOLO("yolov8n.pt")

def send_alert(animal_name):
    sender_email = "your_email@gmail.com"
    receiver_email = "alert_receiver@gmail.com"
    password = "your_email_password"
    
    subject = "Wild Animal Detected!"
    body = f"Alert! A {animal_name} has been detected in the monitored area. Please take necessary precautions."
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = receiver_email
    
    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, msg.as_string())
        server.quit()
        print("Alert email sent successfully!")
    except Exception as e:
        print(f"Error sending email: {e}")

def detect_animals():
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model(frame)  # Perform detection
        
        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])
                label = model.names[cls]
                if label in ["elephant", "tiger", "lion", "bear", "leopard"]:  # Define wild animals
                    send_alert(label)
                    
                    # Draw bounding box
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    cap.release()
    cv2.destroyAllWindows()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(detect_animals(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)

