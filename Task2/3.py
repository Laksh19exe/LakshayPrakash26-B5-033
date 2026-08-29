import os
import cv2
import numpy as np

# ============================================================
# PATHS
# ============================================================

BASE = os.path.dirname(os.path.abspath(__file__))
INPUT = os.path.join(BASE, "input", "3.png")
OUTPUT_DIR = os.path.join(BASE, "output")
OUTPUT = os.path.join(OUTPUT_DIR, "3.jpeg")

# ============================================================
# SETTINGS
# ============================================================

CYAN = (255, 255, 0)       # Boundary line color (BGR)
FILL = (180, 80, 40)       # Blue lane overlay (BGR)
ALPHA = 0.35               # Fill transparency

def generate_curve(control_points, num_points=50):
    """Fits a quadratic polynomial through control points (x, y) to create a smooth curve."""
    pts = np.array(control_points, dtype=np.float32)
    ys = pts[:, 1]
    xs = pts[:, 0]
    
    # Fit quadratic polynomial x = f(y)
    poly = np.polyfit(ys, xs, 2)
    
    # Generate dense curve points from the top of the line down to the bottom
    dense_y = np.linspace(ys.min(), ys.max(), num_points)
    dense_x = np.polyval(poly, dense_y)
    
    return np.column_stack((dense_x, dense_y)).astype(np.int32)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    img = cv2.imread(INPUT)
    if img is None:
        print(f"ERROR: Could not load {INPUT}")
        return

    h, w = img.shape[:2]

    # Y-bounds covering the visible road
    y_top = int(h * 0.40)
    y_mid = int(h * 0.60)
    y_bot = int(h * 0.92)

    # 1. Left boundary curve (Double yellow line)
    # Pulled IN to the right (higher X values) to stop encroaching on the other lane
    left_control = [
        [w * 0.50, y_top],   
        [w * 0.297, y_mid],   
        [w * 0.10, y_bot]    
    ]

    # 2. Right boundary curve (Solid white line)
    # Pulled IN to the left (lower X values) to keep it out of the grass
    right_control = [
        [w * 0.58, y_top],   
        [w * 0.650, y_mid],   
        [w * 0.830, y_bot]    
    ]

    # Generate smooth fitted curves
    left_line = generate_curve(left_control)
    right_line = generate_curve(right_control)

    # 3. Form closed polygon connecting left curve down and right curve back up
    polygon = np.vstack([left_line, right_line[::-1]])

    # 4. Render blue fill overlay
    overlay = img.copy()
    cv2.fillPoly(overlay, [polygon], FILL)
    result = cv2.addWeighted(overlay, ALPHA, img, 1 - ALPHA, 0)

    # 5. Draw cyan line highlights
    cv2.polylines(result, [left_line], isClosed=False, color=CYAN, thickness=4, lineType=cv2.LINE_AA)
    cv2.polylines(result, [right_line], isClosed=False, color=CYAN, thickness=4, lineType=cv2.LINE_AA)

    # Save output
    cv2.imwrite(OUTPUT, result)
    print("Lane detection for 3.png complete.")
    print("Output saved to:", OUTPUT)

if __name__ == "__main__":
    main()