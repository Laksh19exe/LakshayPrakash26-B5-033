import os
import cv2
import numpy as np

# ============================================================
# PATHS
# ============================================================

BASE = os.path.dirname(os.path.abspath(__file__))
INPUT = os.path.join(BASE, "input", "7.jpeg")
OUTPUT_DIR = os.path.join(BASE, "output")
OUTPUT = os.path.join(OUTPUT_DIR, "7.jpeg")

# ============================================================
# SETTINGS
# ============================================================

CYAN = (255, 255, 0)       # Boundary line color (BGR)
FILL = (180, 80, 40)       # Blue lane overlay (BGR)
ALPHA = 0.35               # Fill transparency

def generate_curve(control_points, num_points=50):
    """Fits a quadratic polynomial through control points (x, y) to match the red markers."""
    pts = np.array(control_points, dtype=np.float32)
    ys = pts[:, 1]
    xs = pts[:, 0]
    
    poly = np.polyfit(ys, xs, 2)
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

    # Y-bounds tailored to match the span of the red dashed lines in 7.jpeg
    y_top = int(h * 0.42)
    y_mid = int(h * 0.70)
    y_bot = int(h * 0.90)

    # 1. Left boundary curve (Tracking the solid white shoulder line marked in red)
    left_control = [
        [w * 0.462, y_top],   # Vanishing point apex at the horizon
        [w * 0.35, y_mid],   # Mid point
        [w * 0.27, y_bot]    # Bottom-left edge
    ]

    # 2. Right boundary curve (Tracking the dashed lane divider line marked in red)
    right_control = [
        [w * 0.512, y_top],   # Vanishing point apex at the horizon
        [w * 0.58, y_mid],   # Mid point tracking the dashed line
        [w * 0.64, y_bot]    # Bottom-right edge
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

    # 5. Draw cyan line highlights matching the user's red sketch
    cv2.polylines(result, [left_line], isClosed=False, color=CYAN, thickness=4, lineType=cv2.LINE_AA)
    cv2.polylines(result, [right_line], isClosed=False, color=CYAN, thickness=4, lineType=cv2.LINE_AA)

    # Save output
    cv2.imwrite(OUTPUT, result)
    print("Lane detection for 7.jpeg complete. Saved to:", OUTPUT)

if __name__ == "__main__":
    main()