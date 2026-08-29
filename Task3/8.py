import os
import cv2
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
INPUT = os.path.join(BASE, "input", "8.png")
OUTPUT = os.path.join(BASE, "output", "8.png")

os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

img = cv2.imread(INPUT)

if img is None:
    print("Could not load 8.png")
    exit()

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# =========================================================
# POTHOLES: WHITE CIRCULAR BLOBS
# =========================================================

white = cv2.inRange(
    hsv,
    np.array([0, 0, 180]),
    np.array([180, 80, 255])
)

contours, _ = cv2.findContours(
    white,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

potholes = []

for c in contours:

    area = cv2.contourArea(c)

    if 100 < area < 35000:

        perimeter = cv2.arcLength(c, True)

        if perimeter == 0:
            continue

        circularity = (
            4 * np.pi * area /
            (perimeter * perimeter)
        )

        x, y, w, hh = cv2.boundingRect(c)

        if (
            circularity > 0.15
            and w > 10
            and hh > 5
            and w / hh < 6
            and hh / w < 6
        ):
            potholes.append((x, y, w, hh))

potholes = sorted(potholes, key=lambda box: box[0])


# =========================================================
# OBSTACLES: BLUE CYLINDERS DETECTION
# =========================================================

blue = cv2.inRange(
    hsv,
    np.array([100, 150, 50]),
    np.array([140, 255, 255])
)

contours, _ = cv2.findContours(
    blue,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

obstacles = []

for c in contours:

    area = cv2.contourArea(c)

    if 30 < area < 200000:

        x, y, w, hh = cv2.boundingRect(c)

        if w > 5 and hh > 5:
            obstacles.append((x, y, w, hh))

obstacles = sorted(obstacles, key=lambda box: box[0])


# =========================================================
# DRAW POTHOLES
# =========================================================

for i, (x, y, w, hh) in enumerate(potholes, 1):

    cv2.rectangle(
        img,
        (x, y),
        (x + w, y + hh),
        (255, 0, 255),
        3
    )

    text_x = max(10, x - 20)
    text_y = y - 10 if y > 25 else y + hh + 25

    cv2.putText(
        img,
        f"Pothole {i} ({x},{y})",
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 0, 255),
        2
    )


# =========================================================
# DRAW OBSTACLES (OBSTACLE 2 TEXT MOVED TO THE RIGHT)
# =========================================================

for i, (x, y, w, hh) in enumerate(obstacles, 1):

    cv2.rectangle(
        img,
        (x, y),
        (x + w, y + hh),
        (0, 255, 255),
        3
    )

    if i == 1:
        text_x = x - 40
        text_y = y + hh + 30
    elif i == 2:
        # Moved Obstacle 2 text to the right side of the cylinder into open space
        text_x = x + w + 15
        text_y = y + (hh // 2) + 5
    elif i == 3:
        text_x = x - 70
        text_y = y - 15
    else:
        text_x = x - 70
        text_y = y - 15

    cv2.putText(
        img,
        f"Obstacle {i} ({x},{y})",
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2
    )


# =========================================================
# TOTAL COUNTS (BOTTOM RIGHT)
# =========================================================

img_height, img_width = img.shape[:2]

right_x = img_width - 250
start_y = img_height - 60

cv2.putText(
    img,
    f"Potholes: {len(potholes)}",
    (right_x, start_y),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    (255, 0, 255),
    2
)

cv2.putText(
    img,
    f"Obstacles: {len(obstacles)}",
    (right_x, start_y + 35),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    (0, 255, 255),
    2
)


# =========================================================
# SAVE + DISPLAY
# =========================================================

cv2.imwrite(OUTPUT, img)

print("Potholes:", len(potholes))
print("Obstacles:", len(obstacles))
print("Saved:", OUTPUT)