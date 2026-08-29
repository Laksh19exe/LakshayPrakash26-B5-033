import os
import cv2
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
INPUT = os.path.join(BASE, "input", "1.png")
OUTPUT_DIR = os.path.join(BASE, "output")
OUTPUT = os.path.join(OUTPUT_DIR, "1.png")

os.makedirs(OUTPUT_DIR, exist_ok=True)

img = cv2.imread(INPUT)

if img is None:
    print(f"Could not load {INPUT}")
    exit()

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# -------------------------------------------------
# Detect white pothole
# -------------------------------------------------

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

    if 300 < area < 10000:
        perimeter = cv2.arcLength(c, True)

        if perimeter == 0:
            continue

        circularity = 4 * np.pi * area / (perimeter ** 2)
        x, y, w, hh = cv2.boundingRect(c)

        if circularity > 0.55 and w > 20 and hh > 10:
            potholes.append((x, y, w, hh))

# -------------------------------------------------
# Detect yellow obstacle
# -------------------------------------------------

yellow = cv2.inRange(
    hsv,
    np.array([15, 100, 60]),
    np.array([40, 255, 255])
)

contours, _ = cv2.findContours(
    yellow,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

obstacles = []

for c in contours:
    area = cv2.contourArea(c)

    if 500 < area < 30000:
        x, y, w, hh = cv2.boundingRect(c)

        if w > 20 and hh > 20:
            obstacles.append((x, y, w, hh))

# -------------------------------------------------
# Draw potholes
# -------------------------------------------------

for i, (x, y, w, hh) in enumerate(potholes, 1):
    cv2.rectangle(
        img,
        (x, y),
        (x + w, y + hh),
        (255, 0, 255),
        3
    )

    cv2.putText(
        img,
        f"Pothole {i} ({x},{y})",
        (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 0, 255),
        2
    )

# -------------------------------------------------
# Draw obstacles
# -------------------------------------------------

for i, (x, y, w, hh) in enumerate(obstacles, 1):
    cv2.rectangle(
        img,
        (x, y),
        (x + w, y + hh),
        (0, 255, 255),
        3
    )

    cv2.putText(
        img,
        f"Obstacle {i} ({x},{y})",
        (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2
    )

# -------------------------------------------------
# Counts
# -------------------------------------------------

cv2.putText(
    img,
    f"Potholes: {len(potholes)}",
    (20, 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    (255, 0, 255),
    2
)

cv2.putText(
    img,
    f"Obstacles: {len(obstacles)}",
    (20, 60),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    (0, 255, 255),
    2
)

cv2.imwrite(OUTPUT, img)

print("Potholes:", len(potholes))
print("Obstacles:", len(obstacles))
print("Saved:", OUTPUT)
