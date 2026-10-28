import os
import cv2
import numpy as np
import base64
import io
from flask import Flask, request, jsonify
from PIL import Image, ImageOps

# ==========================================
# AI BACKGROUND REMOVAL (Lazy Loaded)
# ==========================================
REMBG_AVAILABLE = False
rembg_session = None

try:
    from rembg import remove as rembg_remove, new_session
    REMBG_AVAILABLE = True
except Exception as e:
    print(f"rembg import error: {e}")

def get_rembg_session():
    """Server start par model load nahi hoga; pehli image request aane par hoga."""
    global rembg_session
    if rembg_session is None and REMBG_AVAILABLE:
        try:
            # u2netp lightweight model (~4.7 MB) hai, standard u2net (~176 MB)
            # Render ke 512MB RAM free tier ke liye u2netp best hai taaki OOM na ho.
            rembg_session = new_session("u2netp")
        except Exception as e:
            print(f"Failed to initialize rembg session: {e}")
    return rembg_session

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# ==========================================
# OPENCV DOCUMENT PROCESSING FUNCTIONS
# ==========================================

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def _quad_from_contour(cnt):
    hull = cv2.convexHull(cnt)
    peri = cv2.arcLength(hull, True)
    for eps in np.linspace(0.01, 0.10, 25):
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype("float32")
    return cv2.boxPoints(cv2.minAreaRect(hull)).astype("float32")

def _quad_is_valid(quad, w, h):
    area = cv2.contourArea(quad.astype("float32"))
    frac = area / float(w * h)
    if frac < 0.08 or frac > 0.97:
        return False
    return cv2.isContourConvex(quad.reshape(-1, 1, 2).astype("int32"))

def detect_corners_ai(image):
    if not REMBG_AVAILABLE:
        return None
    try:
        session = get_rembg_session()
        if session is None:
            return None

        orig_h, orig_w = image.shape[:2]
        scale = 600.0 / max(orig_h, orig_w)
        if scale < 1:
            small = cv2.resize(image, (int(orig_w * scale), int(orig_h * scale)))
        else:
            small = image
            scale = 1.0

        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        result = rembg_remove(Image.fromarray(rgb), session=session)
        mask = np.array(result)[:, :, 3]

        _, binm = cv2.threshold(mask, 100, 255, cv2.THRESH_BINARY)
        sh, sw = small.shape[:2]
        k = max(5, int(min(sh, sw) * 0.04)) | 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        cleaned = cv2.morphologyEx(binm, cv2.MORPH_OPEN, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < (sw * sh * 0.05):
            return None

        quad = _quad_from_contour(largest)

        center = quad.mean(axis=0)
        pad = k * 0.3
        padded = []
        for (x, y) in quad:
            dx, dy = x - center[0], y - center[1]
            dist = np.hypot(dx, dy) or 1
            padded.append([x + dx / dist * pad, y + dy / dist * pad])
        padded = np.array(padded, dtype="float32")
        padded[:, 0] = np.clip(padded[:, 0], 0, sw - 1)
        padded[:, 1] = np.clip(padded[:, 1], 0, sh - 1)

        if not _quad_is_valid(padded, sw, sh):
            return None

        pts = padded / scale
        pts[:, 0] = np.clip(pts[:, 0], 0, orig_w - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, orig_h - 1)
        return pts.tolist()
    except Exception as e:
        print(f"AI corner detection fallback: {e}")
        return None

def detect_corners(image):
    oh, ow = image.shape[:2]
    scale = 800.0 / max(oh, ow)
    small = cv2.resize(image, (int(ow * scale), int(oh * scale))) if scale < 1 else image.copy()
    if scale >= 1:
        scale = 1.0
    sh, sw = small.shape[:2]

    blur = cv2.GaussianBlur(small, (7, 7), 0)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    masks = []
    _, m1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    masks.append(m1)
    masks.append(cv2.inRange(hsv, (0, 0, 120), (180, 70, 255)))
    masks.append(cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 101, -15))
    edges = cv2.Canny(gray, 30, 150)
    masks.append(cv2.morphologyEx(edges, cv2.MORPH_CLOSE,
                                  cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))))

    k = max(7, int(min(sh, sw) * 0.03)) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    best, best_score = None, 0
    for m in masks:
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=2)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        for c in sorted(contours, key=cv2.contourArea, reverse=True)[:3]:
            quad = _quad_from_contour(c)
            if not _quad_is_valid(quad, sw, sh):
                continue
            q_area = cv2.contourArea(quad)
            fill = cv2.contourArea(c) / (q_area + 1e-6)
            score = q_area * min(fill, 1.0)
            if score > best_score:
                best, best_score = quad, score

    if best is None:
        return [[0, 0], [ow, 0], [ow, oh], [0, oh]]

    pts = best / scale
    pts[:, 0] = np.clip(pts[:, 0], 0, ow - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, oh - 1)
    return pts.tolist()

def auto_orient(image, category):
    h, w = image.shape[:2]
    if category == 'multi':
        if h > w:
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif category == 'single':
        if w > h:
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    return image

def process_enhancement(image, mode):
    if mode == 'clean':
        alpha = 1.1
        beta = 10
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    elif mode == 'bw':
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 3)
        bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 10)
        return cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
    return image

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def base64_to_cv2(b64_string):
    header, encoded = b64_string.split(",", 1)
    img_data = base64.b64decode(encoded)
    np_arr = np.frombuffer(img_data, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR), header

def cv2_to_base64(img, header="data:image/jpeg;base64"):
    _, buffer = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    encoded = base64.b64encode(buffer).decode('utf-8')
    return f"{header},{encoded}"

# ==========================================
# FLASK API ENDPOINTS
# ==========================================
@app.route('/api/upload', methods=['POST'])
def api_upload():
    try:
        if 'image' not in request.files:
            return jsonify({"success": False, "error": "No image file provided."})

        file = request.files['image']
        img_bytes = file.read()

        pil_img = Image.open(io.BytesIO(img_bytes))
        pil_img = ImageOps.exif_transpose(pil_img).convert("RGB")
        cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        corners = detect_corners_ai(cv_img)
        detection_method = "ai"
        if corners is None:
            corners = detect_corners(cv_img)
            detection_method = "classic"

        original_b64 = cv2_to_base64(cv_img)
        height, width = cv_img.shape[:2]

        return jsonify({
            "success": True, "original_b64": original_b64, "width": width, "height": height,
            "corners": corners, "detection_method": detection_method
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/crop', methods=['POST'])
def api_crop():
    try:
        data = request.json
        img, header = base64_to_cv2(data['image'])
        corners = np.array(data['corners'], dtype="float32")
        cropped = four_point_transform(img, corners)
        oriented = auto_orient(cropped, data.get('category', 'single'))
        enhanced = process_enhancement(oriented, data.get('mode', 'original'))
        return jsonify({"success": True, "result_b64": cv2_to_base64(enhanced, header)})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/rotate', methods=['POST'])
def api_rotate():
    try:
        data = request.json
        img, header = base64_to_cv2(data['image'])
        if data['direction'] == 'cw':
            rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif data['direction'] == 'ccw':
            rotated = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            rotated = img
        return jsonify({"success": True, "result_b64": cv2_to_base64(rotated, header)})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

# ==========================================
# FRONTEND HTML/CSS/JS
# ==========================================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DocScan</title>
    <script>
    document.addEventListener('contextmenu', event => event.preventDefault());
    document.addEventListener('keydown', function(e) {
        if (e.keyCode == 123) return false;
        if (e.ctrlKey && e.shiftKey && e.keyCode == 'I'.charCodeAt(0)) return false;
        if (e.ctrlKey && e.shiftKey && e.keyCode == 'C'.charCodeAt(0)) return false;
        if (e.ctrlKey && e.shiftKey && e.keyCode == 'J'.charCodeAt(0)) return false;
        if (e.ctrlKey && e.keyCode == 'U'.charCodeAt(0)) return false;
    });
    </script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        :root {
            --primary: #2563EB; --secondary: #4F46E5; --dark: #0F172A;
            --bg: #F8FAFC; --success: #10B981; --danger: #EF4444;
            --gray-light: #E2E8F0; --gray-text: #64748B;
            --font: 'Inter', system-ui, -apple-system, sans-serif;
            --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
            --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
            --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1);
        }

        * { box-sizing: border-box; margin: 0; padding: 0; font-family: var(--font); }
        body { background-color: var(--bg); color: var(--dark); min-height: 100vh; display: flex; flex-direction: column; }

        header {
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            color: white; padding: 1.2rem 2.5rem;
            box-shadow: var(--shadow-md);
            display: flex; justify-content: space-between; align-items: center;
            position: sticky; top: 0; z-index: 50;
        }
        .logo { font-size: 1.5rem; font-weight: 700; display: flex; align-items: center; gap: 0.75rem; letter-spacing: -0.5px;}

        .main-container { display: flex; flex: 1; overflow: hidden; flex-direction: column; }
        @media (min-width: 768px) { .main-container { flex-direction: row; } }

        .sidebar { background: white; width: 100%; border-bottom: 1px solid var(--gray-light); padding: 2rem; overflow-y: auto; box-shadow: var(--shadow-sm); z-index: 10;}
        @media (min-width: 768px) { .sidebar { width: 340px; border-right: 1px solid var(--gray-light); border-bottom: none; height: calc(100vh - 70px); } }

        .form-group { margin-bottom: 1.5rem; }
        label { display: block; font-weight: 600; margin-bottom: 0.5rem; color: #334155; font-size: 0.95rem;}
        select {
            width: 100%; padding: 0.85rem 1rem; border-radius: 10px;
            border: 1px solid #CBD5E1; font-size: 1rem;
            outline: none; transition: all 0.2s; background: #fff;
            box-shadow: var(--shadow-sm); cursor: pointer;
        }
        select:focus { border-color: var(--primary); box-shadow: 0 0 0 4px rgba(37,99,235,0.15); }

        .btn {
            background: white; border: 1px solid #CBD5E1; cursor: pointer;
            transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
            font-weight: 600; display: inline-flex; align-items: center;
            justify-content: center; gap: 0.5rem; color: #334155;
            padding: 0.75rem 1rem; border-radius: 10px; font-size: 0.95rem;
        }
        .btn:hover { background: #F1F5F9; transform: translateY(-1px); box-shadow: var(--shadow-sm); }
        .btn-primary { background: var(--primary); color: white; border: none; }
        .btn-success { background: var(--success); color: white; border: none; }
        .btn-success:hover { background: #059669; }

        #printAllBtn {
            display: none; width: 100%; margin-top: 2rem; padding: 1.25rem;
            font-size: 1.15rem; font-weight: 700; border-radius: 12px;
            background: linear-gradient(135deg, #10B981, #059669);
            color: white; border: none; box-shadow: 0 4px 15px rgba(16, 185, 129, 0.4);
            transition: transform 0.2s, box-shadow 0.2s;
        }
        #printAllBtn:hover { transform: translateY(-3px); box-shadow: 0 8px 25px rgba(16, 185, 129, 0.5); }

        .workspace { flex: 1; padding: 2.5rem; overflow-y: auto; display: flex; flex-direction: column; align-items: center; gap: 2rem; position: relative; background: #F1F5F9;}

        .empty-state { text-align: center; color: var(--gray-text); margin-top: 10vh; max-width: 400px; }
        .empty-state h2 { color: var(--dark); margin-bottom: 1rem; font-size: 1.8rem; }
        .empty-state svg { width: 90px; height: 90px; margin-bottom: 1.5rem; color: var(--primary); opacity: 0.9; }

        .upload-zones { display: flex; flex-direction: column; gap: 2.5rem; width: 100%; max-width: 900px; }

        .upload-card {
            background: white; border: 2px dashed #94A3B8; border-radius: 16px;
            padding: 3rem 2rem; text-align: center; position: relative;
            transition: all 0.3s ease; flex: 1; display: flex; flex-direction: column;
            justify-content: center; min-height: 280px; box-shadow: var(--shadow-sm);
        }
        .upload-card:hover { border-color: var(--primary); background: #EFF6FF; transform: translateY(-2px); box-shadow: var(--shadow-md); }
        .upload-card.dragover { border-color: var(--primary); background: #DBEAFE; transform: scale(1.02); }
        .upload-card input[type="file"] { position: absolute; inset: 0; width: 100%; height: 100%; opacity: 0; cursor: pointer; z-index: 10; }
        .upload-card .icon { font-size: 3.5rem; margin-bottom: 1rem; color: var(--primary); }

        .result-card {
            background: white; border-radius: 16px;
            box-shadow: var(--shadow-lg); overflow: hidden;
            width: 100%; max-width: 550px; border: 1px solid #E2E8F0;
            display: none; transition: all 0.3s ease;
        }
        .result-header { background: white; padding: 1.2rem 1.5rem; border-bottom: 1px solid #E2E8F0; font-weight: 700; display: flex; justify-content: space-between; align-items: center;}
        .result-image-container {
            width: 100%; background: repeating-conic-gradient(#f8fafc 0% 25%, #f1f5f9 0% 50%) 50% / 20px 20px;
            border-bottom: 1px solid #E2E8F0;
        }
        .result-image { width: 100%; height: auto; display: block; max-height: 450px; object-fit: contain; }

        .result-actions { padding: 1.5rem; display: grid; grid-template-columns: 1fr 1fr; gap: 0.75rem; background: #F8FAFC;}
        .result-actions .full-width { grid-column: span 2; }

        .loader-overlay { position: fixed; inset: 0; background: rgba(15, 23, 42, 0.7); display: none; justify-content: center; align-items: center; z-index: 100; backdrop-filter: blur(5px); }
        .loader-box { background: white; padding: 2.5rem; border-radius: 16px; text-align: center; box-shadow: var(--shadow-lg); }
        .spinner { border: 4px solid #E2E8F0; border-top: 4px solid var(--primary); border-radius: 50%; width: 50px; height: 50px; animation: spin 1s linear infinite; margin: 0 auto 1.5rem auto; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }

        .editor-modal { position: fixed; inset: 0; background: rgba(15, 23, 42, 0.95); display: none; flex-direction: column; z-index: 200; backdrop-filter: blur(8px);}
        .editor-header { padding: 1.2rem 2rem; background: #0F172A; color: white; display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #334155;}
        .editor-body { flex: 1; position: relative; display: flex; justify-content: center; align-items: center; overflow: hidden; padding: 2rem; }
        canvas { max-width: 100%; max-height: 100%; box-shadow: 0 0 30px rgba(0,0,0,0.8); touch-action: none; border-radius: 8px;}
        .editor-footer { padding: 1.5rem; background: #0F172A; display: flex; gap: 1rem; justify-content: center; flex-wrap: wrap; border-top: 1px solid #334155;}

        @media print {
            html, body {
                background-color: #ffffff !important;
                background: none !important;
                min-height: 0 !important;
                margin: 0; padding: 0;
            }
            header, .main-container, .loader-overlay, .editor-modal {
                display: none !important;
            }
            #print-area, #print-area * { visibility: visible; }
            #print-area {
                position: relative;
                left: 0; top: 0; width: 100%;
                display: flex; flex-direction: column; align-items: center;
                justify-content: flex-start;
                padding-top: 1cm;
            }
            .print-page {
                width: 100%; display: flex; flex-direction: column; align-items: center;
                gap: 4cm;
            }
            .print-page img {
                width: 12cm; height: 7.5cm; object-fit: fill; border: 1px solid #ddd;
            }
            .single-print { gap: 0; }
            .single-print img {
                width: 19cm !important;
                height: auto !important;
                max-height: 25cm !important;
                object-fit: contain !important;
                border: none !important;
            }
            @page {
                size: A4 portrait;
                margin: 0.5cm 1cm;
            }
        }
    </style>
</head>
<body>

    <header>
        <div class="logo">
            <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M4 22h14a2 2 0 0 0 2-2V7.5L14.5 2H6a2 2 0 0 0-2 2v4"/><polyline points="14 2 14 8 20 8"/><path d="M3 15h6"/><path d="M3 18h6"/></svg>
            DocScan
        </div>
        <div style="font-size: 0.95rem; font-weight: 500; opacity: 0.9; background: rgba(255,255,255,0.2); padding: 0.4rem 0.8rem; border-radius: 20px;">
            Jeet
        </div>
    </header>

    <div class="main-container">
        <div class="sidebar">
            <div class="form-group">
                <label>Document Category</label>
                <select id="docCategory" onchange="handleCategoryChange()">
                    <option value="single">Single Document (Marksheet, Certificate)</option>
                    <option value="multi">Front + Back ID (Aadhaar, PAN, Voter)</option>
                </select>
            </div>

            <div class="form-group">
                <label>Document Type</label>
                <select id="docType"></select>
            </div>
        </div>

        <div class="workspace">
            <div class="empty-state" id="emptyState">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M3 15v4c0 1.1.9 2 2 2h14a2 2 0 0 0 2-2v-4M17 8l-5-5-5 5M12 3v12"/></svg>
                <h2>Ready to Scan</h2>
                <p>Select your document type. Upload, Drag & Drop, or <b>Paste (Ctrl+V)</b> your image.</p>
            </div>

            <div class="upload-zones" id="uploadZones" style="display: none;">
                <div style="display: flex; gap: 2rem; flex-wrap: wrap; width: 100%; justify-content: center;">

                    <!-- Front Container -->
                    <div style="flex: 1; min-width: 320px; max-width: 500px; display: flex; flex-direction: column; gap: 1rem;">
                        <h3 id="frontTitle" style="text-align: center; color: #475569; font-weight: 700;">DOCUMENT IMAGE</h3>

                        <div class="upload-card" id="frontUpload">
                            <input type="file" accept="image/*" capture="environment" onchange="handleFileInput(event, 'front')">
                            <div class="icon">📄</div>
                            <h3 style="color: var(--dark); font-size: 1.25rem;">Drag, Drop or Paste here</h3>
                            <p style="color: var(--gray-text); margin-top: 0.5rem;">or click to browse your files</p>
                        </div>

                        <div class="result-card" id="frontResult">
                            <div class="result-header">
                                <span id="frontResultTitle">Result</span>
                                <span style="color: var(--success); font-size: 0.9rem; background: #D1FAE5; padding: 0.2rem 0.6rem; border-radius: 12px;" id="frontStatus">Original</span>
                            </div>
                            <div class="result-image-container">
                                <img src="" class="result-image" id="frontImgPreview">
                            </div>
                            <div class="result-actions">
                                <button class="btn btn-success full-width" onclick="performCrop('front')">
                                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M6.13 1L6 16a2 2 0 0 0 2 2h15"/><path d="M1 6.13L16 6a2 2 0 0 1 2 2v15"/></svg> Re-Crop
                                </button>
                                <button class="btn full-width" onclick="openEditor('front')">Manual Corner Adjust</button>

                                <select class="btn" style="text-align: left;" onchange="applyEnhancement('front', this.value)">
                                    <option value="original">Color (Original)</option>
                                    <option value="clean">Color (Enhanced)</option>
                                    <option value="bw">Black & White</option>
                                </select>
                                <button class="btn" onclick="rotateImage('front')">↻ Rotate 90°</button>
                            </div>
                        </div>
                    </div>

                    <!-- Back Container -->
                    <div id="backContainer" style="flex: 1; min-width: 320px; max-width: 500px; display: none; flex-direction: column; gap: 1rem;">
                        <h3 style="text-align: center; color: #475569; font-weight: 700;">BACK SIDE</h3>

                        <div class="upload-card" id="backUpload">
                            <input type="file" accept="image/*" capture="environment" onchange="handleFileInput(event, 'back')">
                            <div class="icon">📄</div>
                            <h3 style="color: var(--dark); font-size: 1.25rem;">Drag, Drop or Paste here</h3>
                            <p style="color: var(--gray-text); margin-top: 0.5rem;">or click to browse your files</p>
                        </div>

                        <div class="result-card" id="backResult">
                            <div class="result-header">
                                <span id="backResultTitle">Back Side</span>
                                <span style="color: var(--success); font-size: 0.9rem; background: #D1FAE5; padding: 0.2rem 0.6rem; border-radius: 12px;" id="backStatus">Original</span>
                            </div>
                            <div class="result-image-container">
                                <img src="" class="result-image" id="backImgPreview">
                            </div>
                            <div class="result-actions">
                                <button class="btn btn-success full-width" onclick="performCrop('back')">
                                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M6.13 1L6 16a2 2 0 0 0 2 2h15"/><path d="M1 6.13L16 6a2 2 0 0 1 2 2v15"/></svg> Re-Crop
                                </button>
                                <button class="btn full-width" onclick="openEditor('back')">Manual Corner Adjust</button>

                                <select class="btn" style="text-align: left;" onchange="applyEnhancement('back', this.value)">
                                    <option value="original">Color (Original)</option>
                                    <option value="clean">Color (Enhanced)</option>
                                    <option value="bw">Black & White</option>
                                </select>
                                <button class="btn" onclick="rotateImage('back')">↻ Rotate 90°</button>
                            </div>
                        </div>
                    </div>
                </div>

                <div style="display: flex; justify-content: center; width: 100%;">
                    <button id="printAllBtn" onclick="printAll()">PRINT DOCUMENT</button>
                </div>
            </div>
        </div>
    </div>

    <div id="print-area"></div>

    <div class="loader-overlay" id="loader">
        <div class="loader-box">
            <div class="spinner"></div>
            <h3 id="loaderText" style="color: var(--dark); font-weight: 600;">Processing...</h3>
        </div>
    </div>

    <div class="editor-modal" id="editorModal">
        <div class="editor-header">
            <h3 style="font-weight: 600;">Manual Corner Adjustment</h3>
            <button class="btn" onclick="closeEditor()" style="width: auto; padding: 0.5rem 1rem; border: none; background: rgba(255,255,255,0.1); color: white;">✕ Cancel</button>
        </div>
        <div class="editor-body" id="editorBody">
            <canvas id="cropCanvas"></canvas>
        </div>
        <div class="editor-footer">
            <button class="btn" onclick="resetCorners()">↺ Reset Corners</button>
            <button class="btn btn-success" style="width: 250px; font-size: 1.1rem;" onclick="applyEditor()">✓ Apply Changes</button>
        </div>
    </div>

    <script>
        const state = {
            front: { originalBase64: null, corners: null, resultBase64: null, mode: 'original', width: 0, height: 0, detectionMethod: 'classic' },
            back: { originalBase64: null, corners: null, resultBase64: null, mode: 'original', width: 0, height: 0, detectionMethod: 'classic' }
        };

        let currentEditSide = null;
        let canvasScale = 1;
        let dragCornerIndex = -1;

        const singleTypes = ["Marksheet", "Certificate", "Transfer Certificate", "Domicile Certificate", "Caste Certificate", "Income Certificate", "Other Document"];
        const multiTypes = ["Aadhaar Card", "PAN Card", "Voter ID", "Driving Licence", "Other ID"];

        function handleCategoryChange() {
            const cat = document.getElementById('docCategory').value;
            const typeSelect = document.getElementById('docType');
            typeSelect.innerHTML = '';

            const options = cat === 'single' ? singleTypes : multiTypes;
            options.forEach(opt => {
                let el = document.createElement('option');
                el.value = opt; el.innerText = opt;
                typeSelect.appendChild(el);
            });

            document.getElementById('emptyState').style.display = 'none';
            document.getElementById('uploadZones').style.display = 'flex';

            if (cat === 'single') {
                document.getElementById('frontTitle').innerText = 'DOCUMENT IMAGE';
                document.getElementById('backContainer').style.display = 'none';
            } else {
                document.getElementById('frontTitle').innerText = 'FRONT SIDE';
                document.getElementById('backContainer').style.display = 'flex';
            }
            checkPrintAll();
        }

        handleCategoryChange();

        function checkPrintAll() {
            const cat = document.getElementById('docCategory').value;
            const printBtn = document.getElementById('printAllBtn');

            if (cat === 'multi' && state.front.resultBase64 && state.back.resultBase64) {
                printBtn.style.display = 'block';
                printBtn.innerHTML = 'PRINT FRONT & BACK';
            } else if (cat === 'single' && state.front.resultBase64) {
                printBtn.style.display = 'block';
                printBtn.innerHTML = 'PRINT DOCUMENT';
            } else {
                printBtn.style.display = 'none';
            }
        }

        function showLoader(text) {
            document.getElementById('loaderText').innerText = text;
            document.getElementById('loader').style.display = 'flex';
        }
        function hideLoader() { document.getElementById('loader').style.display = 'none'; }

        async function processFile(file, side) {
            if (!file) return;

            showLoader("Loading Image...");
            const formData = new FormData();
            formData.append('image', file);

            try {
                const res = await fetch('/api/upload', { method: 'POST', body: formData });
                const data = await res.json();

                if (data.success) {
                    state[side].originalBase64 = data.original_b64;
                    state[side].resultBase64 = data.original_b64;
                    state[side].width = data.width;
                    state[side].height = data.height;
                    state[side].corners = data.corners;
                    state[side].mode = 'original';
                    state[side].detectionMethod = data.detection_method || 'classic';

                    document.getElementById(side + 'ImgPreview').src = data.original_b64;
                    document.getElementById(side + 'Upload').style.display = 'none';
                    document.getElementById(side + 'Result').style.display = 'block';
                    document.getElementById(side + 'ResultTitle').innerText = document.getElementById('docType').value + (side==='back'?' (Back)':'');

                    showLoader("Auto Cropping...");
                    await performCrop(side);
                    checkPrintAll();
                } else {
                    alert("Error: " + data.error);
                    hideLoader();
                }
            } catch (err) {
                alert("Upload failed.");
                hideLoader();
            }
        }

        function handleFileInput(event, side) {
            const file = event.target.files[0];
            processFile(file, side);
            event.target.value = '';
        }

        document.addEventListener('paste', function(e) {
            const clipboardItems = (e.clipboardData || e.originalEvent.clipboardData).items;
            let imageFile = null;

            for (let i = 0; i < clipboardItems.length; i++) {
                if (clipboardItems[i].type.indexOf('image') !== -1) {
                    imageFile = clipboardItems[i].getAsFile();
                    break;
                }
            }

            if (imageFile) {
                const cat = document.getElementById('docCategory').value;
                const isFrontFilled = state.front.originalBase64 !== null;
                const isBackFilled = state.back.originalBase64 !== null;

                if (!isFrontFilled) {
                    processFile(imageFile, 'front');
                } else if (cat === 'multi' && !isBackFilled) {
                    processFile(imageFile, 'back');
                } else {
                    if(confirm("An image is already uploaded. Do you want to replace the FRONT image?")) {
                        processFile(imageFile, 'front');
                    }
                }
            }
        });

        async function performCrop(side) {
            showLoader("Cropping...");
            try {
                const res = await fetch('/api/crop', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        image: state[side].originalBase64,
                        corners: state[side].corners,
                        mode: state[side].mode,
                        category: document.getElementById('docCategory').value
                    })
                });
                const data = await res.json();
                if (data.success) {
                    state[side].resultBase64 = data.result_b64;
                    document.getElementById(side + 'ImgPreview').src = data.result_b64;
                    document.getElementById(side + 'Status').innerText =
                        state[side].detectionMethod === 'ai' ? " AI Auto-Cropped" : "✓ Auto-Cropped";
                } else {
                    alert("Crop failed: " + data.error);
                }
            } catch (err) {
                alert("Processing failed.");
            }
            hideLoader();
        }

        async function applyEnhancement(side, mode) {
            state[side].mode = mode;
            await performCrop(side);
        }

        async function rotateImage(side) {
            showLoader("Rotating...");
            try {
                const res = await fetch('/api/rotate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ image: state[side].resultBase64, direction: 'cw' })
                });
                const data = await res.json();
                if (data.success) {
                    state[side].resultBase64 = data.result_b64;
                    state[side].originalBase64 = data.result_b64;

                    let w = state[side].width;
                    let h = state[side].height;
                    state[side].width = h;
                    state[side].height = w;
                    state[side].corners = [ [0,0], [h,0], [h,w], [0,w] ];

                    document.getElementById(side + 'ImgPreview').src = data.result_b64;
                }
            } catch (err) {
                alert("Rotation failed.");
            }
            hideLoader();
        }

        function printAll() {
            const cat = document.getElementById('docCategory').value;
            const printArea = document.getElementById('print-area');

            if(cat === 'multi') {
                printArea.innerHTML = `
                    <div class="print-page">
                        <img src="${state.front.resultBase64}">
                        <img src="${state.back.resultBase64}">
                    </div>
                `;
            } else {
                printArea.innerHTML = `
                    <div class="print-page single-print">
                        <img src="${state.front.resultBase64}">
                    </div>
                `;
            }
            window.print();
        }

        // --- CANVAS EDITOR LOGIC ---
        const canvas = document.getElementById('cropCanvas');
        const ctx = canvas.getContext('2d');
        let editorImg = new Image();

        function openEditor(side) {
            currentEditSide = side;
            document.getElementById('editorModal').style.display = 'flex';
            editorImg.onload = () => { initCanvas(); };
            editorImg.src = state[side].originalBase64;
        }

        function closeEditor() { document.getElementById('editorModal').style.display = 'none'; }

        async function applyEditor() {
            closeEditor();
            await performCrop(currentEditSide);
        }

        function resetCorners() {
            const w = state[currentEditSide].width;
            const h = state[currentEditSide].height;
            state[currentEditSide].corners = [ [0,0], [w,0], [w,h], [0,h] ];
            drawCanvas();
        }

        function initCanvas() {
            const container = document.getElementById('editorBody');
            const cw = container.clientWidth - 40;
            const ch = container.clientHeight - 40;
            const imgRatio = editorImg.width / editorImg.height;
            const containerRatio = cw / ch;

            if (imgRatio > containerRatio) {
                canvas.width = cw;
                canvas.height = cw / imgRatio;
            } else {
                canvas.height = ch;
                canvas.width = ch * imgRatio;
            }

            canvasScale = canvas.width / editorImg.width;
            drawCanvas();
        }

        function drawCanvas() {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(editorImg, 0, 0, canvas.width, canvas.height);

            const corners = state[currentEditSide].corners.map(c => [c[0] * canvasScale, c[1] * canvasScale]);

            ctx.fillStyle = 'rgba(15, 23, 42, 0.7)';
            ctx.beginPath();
            ctx.moveTo(0,0); ctx.lineTo(canvas.width, 0); ctx.lineTo(canvas.width, canvas.height); ctx.lineTo(0, canvas.height); ctx.closePath();
            ctx.moveTo(corners[0][0], corners[0][1]);
            ctx.lineTo(corners[1][0], corners[1][1]);
            ctx.lineTo(corners[2][0], corners[2][1]);
            ctx.lineTo(corners[3][0], corners[3][1]);
            ctx.closePath();
            ctx.fill('evenodd');

            ctx.strokeStyle = '#10B981';
            ctx.lineWidth = 3;
            ctx.beginPath();
            ctx.moveTo(corners[0][0], corners[0][1]);
            for(let i=1; i<4; i++) ctx.lineTo(corners[i][0], corners[i][1]);
            ctx.closePath();
            ctx.stroke();

            ctx.fillStyle = '#10B981';
            corners.forEach(c => {
                ctx.beginPath();
                ctx.arc(c[0], c[1], 12, 0, 2 * Math.PI);
                ctx.fill();
                ctx.strokeStyle = '#fff';
                ctx.lineWidth = 3;
                ctx.stroke();
            });
        }

        function getMousePos(e) {
            const rect = canvas.getBoundingClientRect();
            let clientX = e.clientX; let clientY = e.clientY;
            if (e.touches && e.touches.length > 0) {
                clientX = e.touches[0].clientX; clientY = e.touches[0].clientY;
            }
            return {
                x: (clientX - rect.left) * (canvas.width / rect.width),
                y: (clientY - rect.top) * (canvas.height / rect.height)
            };
        }

        function handlePointerDown(e) {
            e.preventDefault();
            const pos = getMousePos(e);
            const corners = state[currentEditSide].corners.map(c => [c[0] * canvasScale, c[1] * canvasScale]);
            dragCornerIndex = -1;
            for (let i = 0; i < 4; i++) {
                const dist = Math.hypot(pos.x - corners[i][0], pos.y - corners[i][1]);
                if (dist < 40) { dragCornerIndex = i; break; }
            }
        }

        function handlePointerMove(e) {
            if (dragCornerIndex === -1) return;
            e.preventDefault();
            const pos = getMousePos(e);
            const nx = Math.max(0, Math.min(pos.x, canvas.width));
            const ny = Math.max(0, Math.min(pos.y, canvas.height));
            state[currentEditSide].corners[dragCornerIndex] = [nx / canvasScale, ny / canvasScale];
            drawCanvas();
        }

        function handlePointerUp(e) { dragCornerIndex = -1; }

        canvas.addEventListener('mousedown', handlePointerDown);
        canvas.addEventListener('mousemove', handlePointerMove);
        window.addEventListener('mouseup', handlePointerUp);
        canvas.addEventListener('touchstart', handlePointerDown, {passive: false});
        canvas.addEventListener('touchmove', handlePointerMove, {passive: false});
        window.addEventListener('touchend', handlePointerUp);

        ['front', 'back'].forEach(side => {
            const el = document.getElementById(side + 'Upload');
            el.addEventListener('dragover', (e) => { e.preventDefault(); el.classList.add('dragover'); });
            el.addEventListener('dragleave', () => el.classList.remove('dragover'));
            el.addEventListener('drop', () => el.classList.remove('dragover'));
        });
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return HTML_TEMPLATE

if __name__ == '__main__':
    # Local development ke waqt standard 5000 par run hoga
    # Render environment variable PORT provide karta hai
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
