from flask import Flask, Response
import cv2
import mediapipe as mp
import numpy as np
import serial
import time

url = 'http://192.168.252.177:81/stream'

try:
    arduino = serial.Serial('COM3', 9600)
    time.sleep(2)
    print("Conectado al Arduino por COM3")
except:
    arduino = None
    print("No se pudo conectar al Arduino por COM")

app = Flask(__name__)

# Inicializar MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_selfie = mp.solutions.selfie_segmentation
mp_hands = mp.solutions.hands

pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
segmenter = mp_selfie.SelfieSegmentation(model_selection=1)
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Colores de fondo (BGR)
colors = {
    'rosa': (203, 192, 255),
    'amarillo': (135, 206, 235),
    'morado': (180, 130, 255)
}
color_names = list(colors.keys())
selected_color = colors[color_names[0]]  # Color por defecto

# Cálculo de ángulos
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ab = b - a
    bc = c - b
    radians = np.arctan2(bc[1], bc[0]) - np.arctan2(ab[1], ab[0])
    return int(np.abs(np.degrees(radians)) % 360)

def point_in_rect(x, y, rect):
    x1, y1, x2, y2 = rect
    return x1 <= x <= x2 and y1 <= y <= y2

def get_color_rects(w, h):
    rect_w = w // 6
    rect_h = 80
    margin = 20
    start_y = margin
    rects = []
    center_x = w // 2
    offsets = [-rect_w - margin, 0, rect_w + margin]
    for i, offset in enumerate(offsets):
        x1 = center_x + offset - rect_w // 2
        y1 = start_y
        x2 = x1 + rect_w
        y2 = y1 + rect_h
        rects.append((x1, y1, x2, y2))
    return rects

def gen_frames():
    global selected_color
    cap = cv2.VideoCapture(url)
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rects = get_color_rects(w, h)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detectar mano
        hand_results = hands.process(frame_rgb)
        finger_point = None
        if hand_results.multi_hand_landmarks:
            lm = hand_results.multi_hand_landmarks[0].landmark[8]
            finger_point = (int(lm.x * w), int(lm.y * h))
            cv2.circle(frame, finger_point, 10, (0, 255, 0), -1)

        # Ver si el dedo está en algún rectángulo
        if finger_point:
            for i, rect in enumerate(rects):
                if point_in_rect(finger_point[0], finger_point[1], rect):
                    selected_color = colors[color_names[i]]
                    cx = (rect[0] + rect[2]) // 2
                    cy = (rect[1] + rect[3]) // 2
                    cv2.circle(frame, (cx, cy), 15, (0, 255, 0), 3)

        # Segmentación del cuerpo
        segment_results = segmenter.process(frame_rgb)
        mask = segment_results.segmentation_mask > 0.1
        mask = cv2.GaussianBlur(mask.astype(np.float32), (5,5), 0)
        mask_3c = np.dstack([mask] * 3)
        bg_image = np.full(frame.shape, selected_color, dtype=np.uint8)
        frame = np.where(mask_3c, frame, bg_image)

        # Pose
        results = pose.process(frame_rgb)
        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            cuello = (lm[0].x * w, lm[0].y * h)
            hi = (lm[11].x * w, lm[11].y * h)
            ci = (lm[13].x * w, lm[13].y * h)
            mi = (lm[15].x * w, lm[15].y * h)
            hd = (lm[12].x * w, lm[12].y * h)
            cd = (lm[14].x * w, lm[14].y * h)
            md = (lm[16].x * w, lm[16].y * h)

            ang_cuello = 90
            ang_hi = calculate_angle(hi, ci, mi)
            ang_hd = calculate_angle(hd, cd, md)
            ang_ci = ang_hi
            ang_cd = ang_hd

            if arduino:
                cadena = f"{ang_cuello},{ang_hi},{ang_hd},{ang_ci},{ang_cd}\n"
                arduino.write(cadena.encode('utf-8'))

            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Dibujar rectángulos de selección
        for i, rect in enumerate(rects):
            x1, y1, x2, y2 = rect
            color_bgr = colors[color_names[i]]
            overlay = frame.copy()
            alpha = 0.7
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color_bgr, -1)
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
            cv2.putText(frame, color_names[i].capitalize(), ((x1 + x2) // 2 - 40, y2 - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        frame = cv2.resize(frame, (640, 480))
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return '''
    <html>
    <head><title>Imitación con Fondo </title></head>
    <body style="text-align:center; background:#111; color:white;">
        <h1>Imitación en Vivo con Fondo </h1>
        <img src="/video_feed" width="960" height="720" style="border: 4px solid white; border-radius: 10px;">
    </body>
    </html>
    '''

@app.route('/video_feed')

def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
