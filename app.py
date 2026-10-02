import os
import base64
from io import BytesIO
from flask import Flask, request, jsonify
from flask_cors import CORS
import h5py
import numpy as np
from PIL import Image
import gdown

app = Flask(__name__)
CORS(app)

H5_FILE_PATH = "sar_dataset.h5"

# PASTE YOUR GOOGLE DRIVE FILE ID HERE:
GDRIVE_FILE_ID = "https://drive.google.com/drive/folders/1DndwNDz-QLfVFEOfLaaUo3zj0aUMlWu4?usp=drive_link"

def download_h5_from_drive():
    """Downloads the .H5 file from Google Drive if it doesn't exist locally on server."""
    if not os.path.exists(H5_FILE_PATH):
        print("Downloading .H5 dataset from Google Drive...")
        url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
        gdown.download(url, H5_FILE_PATH, quiet=False)
        print("Download complete!")

# Download file when server starts
try:
    download_h5_from_drive()
except Exception as e:
    print(f"Warning: Failed to download .H5 file: {e}")


def query_h5_data(target_lat, target_lng):
    if not os.path.exists(H5_FILE_PATH):
        raise FileNotFoundError(f"File '{H5_FILE_PATH}' not found on server.")

    with h5py.File(H5_FILE_PATH, 'r') as h5f:
        # Read coordinate matrices (adjust key names if your .H5 uses different dataset names)
        lats = h5f['latitude'][:]
        lons = h5f['longitude'][:]

        # Find nearest pixel index (row, col)
        dist = (lats - target_lat) ** 2 + (lons - target_lng) ** 2
        row, col = np.unravel_index(np.argmin(dist), dist.shape)

        matched_lat = float(lats[row, col])
        matched_lng = float(lons[row, col])
        soil_moisture = float(h5f['soil_moisture'][row, col])

        # Extract 50x50 crop slice around selected coordinate
        rows, cols = lats.shape
        r_start, r_end = max(0, row - 25), min(rows, row + 25)
        c_start, c_end = max(0, col - 25), min(cols, col + 25)
        sar_crop = h5f['sar_band'][r_start:r_end, c_start:c_end]

        c_min, c_max = sar_crop.min(), sar_crop.max()
        if c_max - c_min > 0:
            sar_norm = ((sar_crop - c_min) / (c_max - c_min) * 255).astype(np.uint8)
        else:
            sar_norm = np.zeros_like(sar_crop, dtype=np.uint8)

        img = Image.fromarray(sar_norm)
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return matched_lat, matched_lng, soil_moisture, img_b64


@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "status": "online",
        "message": "AI-IoT Dual-Band SAR Server is running!"
    })


@app.route('/get-sar-data', methods=['POST'])
def get_sar_data():
    try:
        data = request.get_json(force=True)
        lat = float(data.get('lat'))
        lng = float(data.get('lng'))

        m_lat, m_lng, moisture, img_b64 = query_h5_data(lat, lng)

        return jsonify({
            "status": "success",
            "lat": lat,
            "lng": lng,
            "matched_lat": f"{m_lat:.5f}",
            "matched_lng": f"{m_lng:.5f}",
            "h5_moisture": f"{moisture:.1f}%",
            "sar_image": f"data:image/png;base64,{img_b64}"
        })
    except FileNotFoundError as fnf_err:
        return jsonify({"status": "error", "message": str(fnf_err)}), 404
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)