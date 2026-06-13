import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cat_ears = cv2.imread('cat_ears.png', cv2.IMREAD_UNCHANGED)

def draw_cat_nose(frame, x, y, w, h):
    # Center of face
    cx = x + w // 2
    cy = y + h // 2

    # Nose triangle (pink)
    nose_size = max(w // 12, 6)
    nose_pts = np.array([
        [cx, cy + nose_size],
        [cx - nose_size, cy],
        [cx + nose_size, cy]
    ])
    cv2.fillPoly(frame, [nose_pts], (147, 112, 219))

    # Whiskers - left side
    wlen = w // 2
    wy   = cy + nose_size // 2
    cv2.line(frame, (cx - nose_size - 5, wy),       (cx - nose_size - wlen, wy - 10), (255, 255, 255), 2)
    cv2.line(frame, (cx - nose_size - 5, wy + 5),   (cx - nose_size - wlen, wy + 10), (255, 255, 255), 2)
    cv2.line(frame, (cx - nose_size - 5, wy + 10),  (cx - nose_size - wlen, wy + 25), (255, 255, 255), 2)

    # Whiskers - right side
    cv2.line(frame, (cx + nose_size + 5, wy),       (cx + nose_size + wlen, wy - 10), (255, 255, 255), 2)
    cv2.line(frame, (cx + nose_size + 5, wy + 5),   (cx + nose_size + wlen, wy + 10), (255, 255, 255), 2)
    cv2.line(frame, (cx + nose_size + 5, wy + 10),  (cx + nose_size + wlen, wy + 25), (255, 255, 255), 2)

def overlay_image(background, overlay, x, y, w, h):
    overlay_resized = cv2.resize(overlay, (w, h))
    x1, x2 = max(x, 0), min(x + w, background.shape[1])
    y1, y2 = max(y, 0), min(y + h, background.shape[0])
    ox1 = x1 - x
    oy1 = y1 - y
    ox2 = ox1 + (x2 - x1)
    oy2 = oy1 + (y2 - y1)
    if x2 <= x1 or y2 <= y1:
        return
    if overlay_resized.shape[2] == 4:
        alpha = overlay_resized[oy1:oy2, ox1:ox2, 3] / 255.0
        for c in range(3):
            background[y1:y2, x1:x2, c] = (
                alpha * overlay_resized[oy1:oy2, ox1:ox2, c] +
                (1 - alpha) * background[y1:y2, x1:x2, c]
            )
    else:
        background[y1:y2, x1:x2] = overlay_resized[oy1:oy2, ox1:ox2]

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) > 0:
        for (x, y, w, h) in faces:
            # Cat ears above face
            ears_w = int(w * 1.3)
            ears_h = int(h * 0.9)
            ears_x = x - int((ears_w - w) / 2)
            ears_y = y - int(ears_h * 0.7)
            overlay_image(frame, cat_ears, ears_x, ears_y, ears_w, ears_h)

            # Cat nose + whiskers drawn directly
            draw_cat_nose(frame, x, y, w, h)

            # Name tag
            cv2.rectangle(frame, (x, y+h), (x+w, y+h+35), (0, 255, 0), -1)
            cv2.putText(frame, "M. Umer Farooq", (x+5, y+h+25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)

        cv2.putText(frame, "CAT FILTER ON", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "FILTER OFF - Cover Face to Hide", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.putText(frame, "AR/VR App - DARVA Lab", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.putText(frame, "Muhammad Umer Farooq", (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)

    cv2.imshow("AR Cat Filter - DARVA Lab", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()