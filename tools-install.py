import os
import sys
import requests
import logging
from tqdm import tqdm

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] tools-install: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("tools-install")

MODELS_DIR = os.path.join(os.getcwd(), "models", "heavy")
os.makedirs(MODELS_DIR, exist_ok=True)

# Model URLs (Direct Links)
MODELS = {
    "RealESRGAN_x4plus.pth": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    "GFPGANv1.4.pth": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
    "parsing_parsenet.pth": "https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth"
}

def download_file(url, dest_path):
    if os.path.exists(dest_path):
        logger.info(f"✅ {os.path.basename(dest_path)} already exists.")
        return True
        
    logger.info(f"📥 Downloading {os.path.basename(dest_path)}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        
        with open(dest_path, 'wb') as f, tqdm(
            desc=os.path.basename(dest_path),
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                bar.update(size)
        return True
    except Exception as e:
        logger.error(f"❌ Failed to download {url}: {e}")
        return False

def main():
    logger.info("🚀 Starting Heavy Engine Model Installer...")
    
    success = True
    for name, url in MODELS.items():
        dest = os.path.join(MODELS_DIR, name)
        if not download_file(url, dest):
            success = False
            
    if success:
        logger.info("✨ All models installed successfully.")
    else:
        logger.error("⚠️ Some models failed to download.")
        sys.exit(1)

if __name__ == "__main__":
    main()
