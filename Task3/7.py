import os
import cv2
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
INPUT = os.path.join(BASE, "input", "7.png")
OUTPUT = os.path.join(BASE, "output", "7.png")

os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

img = cv2.imread(INPUT)

if img is None:
    print("Could not load 7.png")
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

    if 300 < area < 35000:

        perimeter = cv2.arcLength(c, True)

        if perimeter == 0:
            continue

        circularity = (
            4 * np.pi * area /
            (perimeter * perimeter)
        )

        x, y, w, hh = cv2.boundingRect(c)

        if (
            circularity > 0.25
            and w > 20
            and hh > 10
            and w / hh < 6
            and hh / w < 6
        ):
            potholes.append((x, y, w, hh))


# =========================================================
# OBSTACLES: DETECTING ALL OBJECTS
# =========================================================

yellow = cv2.inRange(
    hsv,
    np.array([15, 60, 60]),
    np.array([40, 255, 255])
)

green = cv2.inRange(
    hsv,
    np.array([35, 60, 60]),
    np.array([85, 255, 255])
)

combined_obstacles_mask = cv2.bitwise_or(yellow, green)

contours, _ = cv2.findContours(
    combined_obstacles_mask,
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

# Sort obstacles from left to right across the image
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

    text_y = y + 25 if y < 35 else y - 10

    cv2.putText(
        img,
        f"Pothole {i} ({x},{y})",
        (x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 0, 255),
        2
    )


# =========================================================
# DRAW OBSTACLES (REPOSITIONED TEXT TO THE LEFT)
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
        # Obstacle 1: moved further left as requested
        text_x = x - 200
        text_y = y + 55
    elif i == 2:
        # Obstacle 2: top-left yellow pair #2
        text_x = x - 20
        text_y = y - 15
    elif i == 3:
        # Obstacle 3: small object
        text_x = x - 50
        text_y = y - 25
    elif i == 4:
        # Obstacle 4: moved further to the left outside the green cylinder box
        text_x = x - 170
        text_y = y - 15
    else:
        # Obstacle 5: green cylinder right cluster
        text_x = x - 70
        text_y = y + hh + 30

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