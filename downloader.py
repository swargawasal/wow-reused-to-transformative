import os
import logging
import yt_dlp
import glob
import time
from datetime import datetime
import re

logger = logging.getLogger("downloader")

DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

def _sanitize_filename(name: str) -> str:
    """Sanitize filename."""
    clean = re.sub(r'[^\w\s-]', '', name)
    return clean.replace(' ', '_')

def download_video(url: str, custom_title: str = None) -> str:
    """
    Download video from URL synchronously.
    Strategy:
    1. Check cache (if same title downloaded recently).
    2. Try without cookies.
    3. Retry with cookies.txt (auto-detected or from env).
    4. Retry with Instagram username/password.
    5. Retry with browser cookies (Chrome).
    Returns the absolute path to the downloaded file.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if custom_title:
        clean_title = _sanitize_filename(custom_title)
        filename = f"{clean_title}__{timestamp}.mp4"
        
        # Check cache: if file with same title exists and is recent (<1 hour), use it
        cache_pattern = os.path.join(DOWNLOAD_DIR, f"{clean_title}__*.mp4")
        existing_files = glob.glob(cache_pattern)
        if existing_files:
            # Get most recent file
            existing_files.sort(key=os.path.getmtime, reverse=True)
            latest_file = existing_files[0]
            file_age = time.time() - os.path.getmtime(latest_file)
            
            # If file is less than 1 hour old, use cached version
            if file_age < 3600:
                logger.info(f"⚡ Using cached download: {latest_file}")
                return os.path.abspath(latest_file)
    else:
        filename = f"video__{timestamp}.mp4"
        
    output_path = os.path.join(DOWNLOAD_DIR, filename)
    absolute_path = os.path.abspath(output_path)
    
    # Base options
    ydl_opts = {
        'outtmpl': absolute_path,
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'noplaylist': True,
        'quiet': True,
        'no_warnings': True,
        'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    }

    # Attempt 1: No Cookies
    try:
        logger.info(f"⬇️ Downloading (Attempt 1 - No Cookies): {url}")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
            
        if os.path.exists(absolute_path):
            logger.info(f"✅ Download complete: {absolute_path}")
            return absolute_path
    except Exception as e:
        logger.warning(f"⚠️ Attempt 1 failed: {e}")

    # Attempt 2: With Cookies File
    cookies_path = os.getenv("COOKIES_FILE", "").strip('"').strip("'")
    
    # Auto-detect cookies.txt if not specified
    if not cookies_path and os.path.exists("cookies.txt"):
        cookies_path = "cookies.txt"
        
    if cookies_path and os.path.exists(cookies_path):
        logger.info(f"🔄 Retrying with cookies from file: {cookies_path}")
        ydl_opts['cookiefile'] = cookies_path
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
                
            if os.path.exists(absolute_path):
                logger.info(f"✅ Download complete (with cookies): {absolute_path}")
                return absolute_path
        except Exception as e:
            logger.error(f"❌ Download error (Attempt 2 - File): {e}")

    # Attempt 3: With Username/Password (Instagram)
    ig_username = os.getenv("IG_USERNAME", "").strip()
    ig_password = os.getenv("IG_PASSWORD", "").strip()
    
    if ig_username and ig_password and "instagram.com" in url:
        logger.info("🔄 Retrying with Instagram credentials...")
        ydl_opts.pop('cookiefile', None)
        ydl_opts['username'] = ig_username
        ydl_opts['password'] = ig_password
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
                
            if os.path.exists(absolute_path):
                logger.info(f"✅ Download complete (with credentials): {absolute_path}")
                return absolute_path
        except Exception as e:
            logger.error(f"❌ Download error (Attempt 3 - Credentials): {e}")

    # Attempt 4: With Browser Cookies (Fallback)
    logger.info("🔄 Retrying with cookies from browser (Chrome)...")
    ydl_opts.pop('cookiefile', None)
    ydl_opts.pop('username', None)
    ydl_opts.pop('password', None)
    ydl_opts['cookiesfrombrowser'] = ('chrome',) 
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
            
        if os.path.exists(absolute_path):
            logger.info(f"✅ Download complete (with browser cookies): {absolute_path}")
            return absolute_path
    except Exception as e:
        logger.warning(f"⚠️ Browser cookies failed (Chrome may be running): {e}")

    logger.error("❌ All download attempts failed.")
    logger.info("💡 Tip: For Instagram, either:")
    logger.info("   1. Add cookies.txt file to project root, OR")
    logger.info("   2. Set IG_USERNAME and IG_PASSWORD in .env, OR")
    logger.info("   3. Close Chrome and try again (for browser cookies)")
    return None