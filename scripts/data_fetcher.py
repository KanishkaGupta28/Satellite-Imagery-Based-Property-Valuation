"""
Satellite Image Downloader using Sentinel Hub (OAuth)
----------------------------------------------------

Reads:
- data/train (2).csv
- data/test.csv

Downloads Sentinel-2 RGB satellite images using lat/long
and saves them to:
- images/train/
- images/test/

This script is safe, reproducible, and ML-ready.
"""

import os
import time
import pandas as pd
from tqdm import tqdm
from PIL import Image

from sentinelhub import (
    SHConfig,
    SentinelHubRequest,
    DataCollection,
    MimeType,
    BBox,
    CRS,
    bbox_to_dimensions
)

# ==================================================
# 1. SENTINEL HUB AUTHENTICATION (OAuth)
# ==================================================
config = SHConfig()

# 🔴 PASTE YOUR REAL CREDENTIALS HERE
config.sh_client_id = "7ff57b22-bb0a-4019-b8cc-cf0de747cf3f"
config.sh_client_secret = "kUBtBm7DpV1NNUbrcVaR0MjXjWYmQtFg"

if not config.sh_client_id or not config.sh_client_secret:
    raise RuntimeError("❌ Sentinel Hub Client ID / Secret missing")

# ==================================================
# 2. PATHS & CONSTANTS
# ==================================================
TRAIN_CSV = "data/train (2).csv"
TEST_CSV = "data/test.csv"

TRAIN_IMAGE_DIR = "images/train"
TEST_IMAGE_DIR = "images/test"

os.makedirs(TRAIN_IMAGE_DIR, exist_ok=True)
os.makedirs(TEST_IMAGE_DIR, exist_ok=True)

IMAGE_SIZE = 256        # Final image size
RESOLUTION = 10         # Sentinel-2 resolution (meters)
BBOX_DELTA = 0.0015     # Area around property

# ==================================================
# 3. EVALSCRIPT (TRUE-COLOR RGB)
# ==================================================
evalscript = """
//VERSION=3
function setup() {
  return {
    input: ["B04", "B03", "B02"],
    output: { bands: 3, sampleType: "UINT8" }
  };
}

function evaluatePixel(sample) {
  return [
    sample.B04 * 255,
    sample.B03 * 255,
    sample.B02 * 255
  ];
}
"""

# ==================================================
# 4. IMAGE DOWNLOAD FUNCTION
# ==================================================
def download_images(csv_path, output_dir):
    df = pd.read_csv(csv_path)

    # Validate columns
    if not {"lat", "long"}.issubset(df.columns):
        raise ValueError("CSV must contain 'lat' and 'long' columns")

    # Create ID column if missing
    if "id" not in df.columns:
        df["id"] = range(len(df))

    print(f"\n📡 Downloading images from {csv_path}")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        pid = row["id"]
        lat = row["lat"]
        lon = row["long"]

        img_path = os.path.join(output_dir, f"{pid}.png")
        if os.path.exists(img_path):
            continue

        try:
            bbox = BBox(
                [lon - BBOX_DELTA, lat - BBOX_DELTA,
                 lon + BBOX_DELTA, lat + BBOX_DELTA],
                crs=CRS.WGS84
            )

            size = bbox_to_dimensions(bbox, resolution=RESOLUTION)

            request = SentinelHubRequest(
                evalscript=evalscript,
                input_data=[
                    SentinelHubRequest.input_data(
                        data_collection=DataCollection.SENTINEL2_L2A,
                        time_interval=("2023-01-01", "2023-12-31"),
                        mosaicking_order="leastCC"
                    )
                ],
                responses=[
                    SentinelHubRequest.output_response(
                        "default", MimeType.PNG
                    )
                ],
                bbox=bbox,
                size=size,
                config=config
            )

            # ✅ CRITICAL FIX: NO save_data=True
            image = request.get_data()[0]

            img = Image.fromarray(image)
            img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
            img.save(img_path)

            time.sleep(0.2)  # Avoid rate limiting

        except Exception as e:
            print(f"⚠️ Failed for ID {pid}: {e}")

    print(f"✅ Images saved to {output_dir}")

# ==================================================
# 5. MAIN EXECUTION
# ==================================================
if __name__ == "__main__":
    download_images(TRAIN_CSV, TRAIN_IMAGE_DIR)
    download_images(TEST_CSV, TEST_IMAGE_DIR)

    print("\n🎉 Satellite image download completed successfully!")
