from PIL import Image
import io
import base64
from flask_cors import CORS
from flask import Flask, request, jsonify
from sklearn.cluster import DBSCAN
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


app = Flask(__name__)
CORS(app)


def line_params(x1, y1, x2, y2):
    """Return slope and intercept of a line, handling vertical lines."""
    if x2 != x1:
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
    else:
        m = None  # vertical line
        b = x1    # use x-intercept
    return m, b


def are_collinear(l1, l2,
                  angle_tol_deg=8,     # angle difference tolerance
                  dist_tol=25,         # max distance between segment and other's infinite line
                  merge_gap=80):       # max gap between segments
    # unpack
    x1, y1, x2, y2 = l1
    x3, y3, x4, y4 = l2

    # Convert to angles
    angle1 = np.degrees(np.arctan2(y2 - y1, x2 - x1))
    angle2 = np.degrees(np.arctan2(y4 - y3, x4 - x3))

    # if angles differ too much → not same line
    if abs(angle1 - angle2) > angle_tol_deg:
        return False

    # Check distance of endpoints of l2 to infinite line of l1
    def point_line_distance(px, py, x1, y1, x2, y2):
        return abs((y2 - y1)*px - (x2 - x1)*py + x2*y1 - y2*x1) / \
            np.sqrt((y2 - y1)**2 + (x2 - x1)**2)

    d1 = point_line_distance(x3, y3, x1, y1, x2, y2)
    d2 = point_line_distance(x4, y4, x1, y1, x2, y2)

    # If both endpoints are far from the line → not collinear
    if d1 > dist_tol and d2 > dist_tol:
        return False

    # Check if the segments are close along the line direction
    # If they are too far apart, don’t merge
    p1 = np.array([[x1, y1], [x2, y2]])
    p2 = np.array([[x3, y3], [x4, y4]])

    dist_ab = np.min(np.linalg.norm(p1[:, None, :] - p2[None, :, :], axis=2))
    if dist_ab > merge_gap:
        return False

    return True


def merge_segments(segments):
    """Merge multiple collinear line segments into one long segment."""
    points = []
    for x1, y1, x2, y2 in segments:
        points.append((x1, y1))
        points.append((x2, y2))

    # Choose endpoints farthest apart
    points = np.array(points)
    dists = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    i, j = np.unravel_index(np.argmax(dists), dists.shape)
    return (*points[i], *points[j])


def merge_clustered_line_segments(points):
    """
    points: Nx2 array of all endpoints in a cluster
    Returns: x1, y1, x2, y2 of merged segment
    """
    points = np.array(points)

    # Fit line using least squares
    [vx, vy, x0, y0] = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
    # vx, vy, x0, y0 = float(vx), float(vy), float(x0), float(y0)
    vx, vy, x0, y0 = float(vx[0]), float(vy[0]), float(x0[0]), float(y0[0])

    # Project all points onto the line
    t = ((points[:, 0] - x0)*vx + (points[:, 1] - y0)*vy)

    # Use min/max projections as endpoints
    t_min, t_max = t.min(), t.max()
    x1, y1 = x0 + vx * t_min, y0 + vy * t_min
    x2, y2 = x0 + vx * t_max, y0 + vy * t_max

    return int(x1), int(y1), int(x2), int(y2)


def merge_lines_aggressive_segments(lines, angle_eps=10, dist_eps=30, min_samples=2):
    if lines is None or len(lines) == 0:
        return []

    # Extract endpoints and compute features
    pts = []
    angle_list = []
    dist_list = []

    for (x1, y1, x2, y2) in [l[0] for l in lines]:
        pts.append((x1, y1, x2, y2))

        # Angle in degrees
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        angle = (angle + 180) % 180  # normalize 0–180
        angle_list.append(angle)

        # Distance from origin (Hesse normal form)
        A = y2 - y1
        B = x1 - x2
        C = x2*y1 - y2*x1
        dist = abs(C) / np.sqrt(A*A + B*B)
        dist_list.append(dist)

    angle_array = np.array(angle_list).reshape(-1, 1)
    dist_array = np.array(dist_list).reshape(-1, 1)

    # Cluster by angle
    angle_labels = DBSCAN(eps=angle_eps, min_samples=1).fit(
        angle_array).labels_

    merged_lines = []

    # For each angle cluster, cluster by distance
    for cluster_id in np.unique(angle_labels):
        idx = np.where(angle_labels == cluster_id)[0]
        dist_sub = dist_array[idx]
        dist_labels = DBSCAN(
            eps=dist_eps, min_samples=min_samples).fit(dist_sub).labels_

        # Merge points within each distance cluster
        for dist_id in np.unique(dist_labels):
            jdx = idx[np.where(dist_labels == dist_id)[0]]

            all_points = []
            for k in jdx:
                x1, y1, x2, y2 = pts[k]
                all_points.append([x1, y1])
                all_points.append([x2, y2])

            # Compute merged segment with real endpoints
            x1, y1, x2, y2 = merge_clustered_line_segments(all_points)
            merged_lines.append((x1, y1, x2, y2))

    return merged_lines


def merge_boxes_nearby(boxes, iou_thresh=0.3, distance_thresh=20):
    """
    Merge overlapping or nearby boxes.

    boxes: list of [x, y, w, h]
    iou_thresh: merge if IoU > threshold
    distance_thresh: merge if boxes are close enough
    """
    boxes = [list(b) for b in boxes]
    merged = []

    while boxes:
        base = boxes.pop(0)
        bx1, by1, bw, bh = base
        bx2, by2 = bx1 + bw, by1 + bh

        i = 0
        while i < len(boxes):
            x, y, w, h = boxes[i]
            xx2, yy2 = x + w, y + h

            # Intersection
            xx1 = max(bx1, x)
            yy1 = max(by1, y)
            xx2i = min(bx2, xx2)
            yy2i = min(by2, yy2)
            w_int = max(0, xx2i - xx1)
            h_int = max(0, yy2i - yy1)
            inter = w_int * h_int

            area1 = bw * bh
            area2 = w * h
            iou = inter / float(area1 + area2 - inter)

            # Distance between box centers
            cx1, cy1 = (bx1 + bx2)/2, (by1 + by2)/2
            cx2, cy2 = (x + xx2)/2, (y + yy2)/2
            dist = np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)

            if iou > iou_thresh or dist < distance_thresh:
                # Merge boxes
                bx1 = min(bx1, x)
                by1 = min(by1, y)
                bx2 = max(bx2, xx2)
                by2 = max(by2, yy2)
                bw, bh = bx2 - bx1, by2 - by1
                boxes.pop(i)
            else:
                i += 1

        merged.append([bx1, by1, bw, bh])
    return merged


@app.route('/', methods=['POST'])
def receive_image():
    data = request.get_json()

    if not data or 'image' not in data:
        return jsonify({'error': 'No image provided'}), 400

    image_data = data['image']
    if ',' in image_data:
        image_data = image_data.split(',')[1]
    img_bytes = base64.b64decode(image_data)
    img_pil = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    img_cv = np.array(img_pil)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

    # ==================== MSER Text Detection ====================
    # filename = 'cv_env/hello1.jpeg'
    # img_cv = cv2.imread(filename)
    # if img_cv is None:
    #     raise FileNotFoundError(f"Cannot find the image file: {filename}")

    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    blur2 = cv2.bilateralFilter(gray, 9, 75, 75)
    edges = cv2.Canny(blur, 100, 200)

    # MSER detector
    mser = cv2.MSER_create()
    mser.setDelta(3)        # sensitivity
    mser.setMinArea(500)    # min area of region
    mser.setMaxArea(1300)   # max area of region
    regions, _ = mser.detectRegions(blur2)
    boxes = [cv2.boundingRect(p.reshape(-1, 1, 2)) for p in regions]

    filtered_boxes = []
    for (x, y, w, h) in boxes:
        aspect_ratio = w / float(h)
        if 0.6 < aspect_ratio < 2:  # adjust based on your text
            filtered_boxes.append((x, y, w, h))
    boxes = filtered_boxes

    boxes = merge_boxes_nearby(boxes, iou_thresh=0.3, distance_thresh=150)

    img_text_boxes = img_cv.copy()
    for (x, y, w, h) in boxes:
        cv2.rectangle(img_text_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)
    print(f"Detected text boxes (MSER): {len(boxes)}")

    # ==================== Hough Line Detection ====================

    # use masked edges from text boxes to avoid detecting lines inside text
    # mask = np.ones(edges.shape, dtype="uint8") * 255
    # for (x, y, w, h) in boxes:
    #     cv2.rectangle(mask, (x, y), (x + w, y + h), 0, -1)
    # masked_edges = cv2.bitwise_and(edges, edges, mask=mask)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=75,
                            minLineLength=100, maxLineGap=30)
    img_lines = img_cv.copy()
    merged_lines = []
    # store all vertical lines
    vertical_lines = []
    # store all horizontal lines
    horizontal_lines = []

    if lines is not None:
        print(f"Number of lines: {len(lines)}")
        merged_lines = merge_lines_aggressive_segments(lines)
        print("Number of merged line segments:", len(merged_lines))

        vertical_count = 0
        horizontal_count = 0
        other_count = 0
        tolerance = 30
        for x1, y1, x2, y2 in merged_lines:
            if abs(x1 - x2) < tolerance:
                label = "Vertical" + str(vertical_count)
                vertical_count += 1
                vertical_lines.append((x1, y1, x2, y2))
            elif abs(y1 - y2) < tolerance:
                label = "Horizontal" + str(horizontal_count)
                horizontal_count += 1
                horizontal_lines.append((x1, y1, x2, y2))
            else:
                continue  # skip Other lines completely

            # Draw line
            color = tuple(random.randint(0, 255) for _ in range(3))
            cv2.line(img_lines, (x1, y1), (x2, y2), color, 5)

            # Draw label
            cv2.putText(img_lines, label, ((x1 + x2) // 2, (y1 + y2) // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    print(f"Vertical lines: {vertical_count}")
    print(f"Horizontal lines: {horizontal_count}")
    print(f"Other lines: {other_count}")

    # ==================== Save combined image ====================

    def save_combined_image(images, titles, save_path='combined.png', dpi=600):
        n = len(images)
        plt.figure(figsize=(5 * n, 5), dpi=dpi)

        for i, (img, title) in enumerate(zip(images, titles), 1):
            plt.subplot(1, n, i)
            plt.title(title)
            if len(img.shape) == 2:  # grayscale
                plt.imshow(img, cmap='gray')
            else:
                plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.axis('off')

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"Saved combined image to {save_path}")

    # ==================== Analyze Vertical Line Connections ====================
    vertical_data = {}
    # for each vertical line, find all the overlapping boxes
    print('================== Vertical to Box Connections ==================')
    for i, (x1, y1, x2, y2) in enumerate(vertical_lines):
        for (bx, by, bw, bh) in boxes:
            # check if line i is within box x range
            if bx <= x1 <= bx + bw:
                if i not in vertical_data:
                    vertical_data[i] = []
                vertical_data[i].append((bx, by, bw, bh))
        # sort boxes by y position
        if i in vertical_data:
            vertical_data[i].sort(key=lambda b: b[1])
        # print the number of boxes connected to this line
        num_boxes = len(vertical_data.get(i, []))
        print(f"Vertical line {i} connects to {num_boxes} text boxes.")
    print(vertical_data)

    # for each horizontal line, find all the overlapping vertical lines
    print('================== Horizontal to Vertical Connections ==================')
    horizontal_data = {}
    tolerance = 20
    for i, (x1, y1, x2, y2) in enumerate(horizontal_lines):
        for j, (vx1, vy1, vx2, vy2) in enumerate(vertical_lines):
            if (min(x1, x2) - tolerance <= vx1 <= max(x1, x2) + tolerance and
                    min(vy1, vy2) - tolerance <= y1 <= max(vy1, vy2) + tolerance):
                if i not in horizontal_data:
                    horizontal_data[i] = []
                horizontal_data[i].append(j)
        num_vlines = len(horizontal_data.get(i, []))
        # print the vertical line indices
        if i in horizontal_data:
            print(f"Vertical lines {i} connect to: {horizontal_data[i]}")
    print(horizontal_data)

    # for each horizontal_data, find the first box connected to the vertical line
    # and that box has to be below the horizontal line
    print('================== Horizontal to Box Connections ==================')
    horizontal_box_connections = {}
    for hline_idx, vline_indices in horizontal_data.items():
        hy1 = horizontal_lines[hline_idx][1]
        connected_boxes = []
        for vline_idx in vline_indices:
            boxes_on_vline = vertical_data.get(vline_idx, [])
            for (bx, by, bw, bh) in boxes_on_vline:
                if by > hy1:  # box is below horizontal line
                    connected_boxes.append((bx, by, bw, bh))
                    break  # only take the first box below the line
        horizontal_box_connections[hline_idx] = connected_boxes
        print(
            f"Horizontal line {hline_idx} first connects to boxes: {connected_boxes}")
    print(horizontal_box_connections)

    images = [img_cv, gray, blur, edges, img_text_boxes, img_lines]
    titles = ['Original', 'Gray', 'Blur', 'Canny',
              'Text Boxes (MSER)', 'Hough Lines']
    save_combined_image(images, titles, 'cv_env/result.png')

    # return vertical_lines, horizontal_lines, vertical_data and horizontal_box_connections
    return jsonify({
        'status_code': 200,
        'vertical_lines': vertical_lines,
        'horizontal_lines': horizontal_lines,
        'vertical_data': vertical_data,
        'horizontal_box_connections': horizontal_box_connections,
        'boxes': boxes
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7500)
