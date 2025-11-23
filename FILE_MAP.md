# Project Structure

```
project/
│
├── downloads/          # Downloaded videos
├── merged_videos/      # Final output videos
├── temp/               # Temporary processing files
├── tools/              # Downloaded tools/binaries
├── models/             # AI model weights
│   └── heavy/          # RealESRGAN & GFPGAN models
├── venv/               # Python virtual environment
├── __pycache__/        # Python cache
│
├── .env                # Environment configuration
├── cookies.txt         # Instagram/social media cookies
├── token.json          # YouTube API token
├── client_secret.json  # Google OAuth credentials
├── upload_log.csv      # Upload history
│
├── main.py             # Bot entry point
├── downloader.py       # Video downloading
├── compiler.py         # Video processing & AI enhancement (includes HeavyEditor)
├── audio_processing.py # Audio remixing
├── copyright_checker.py# Copyright detection
├── uploader.py         # YouTube upload
├── tools-install.py    # Model installer
├── requirements.txt    # Python dependencies
└── FILE_MAP.md         # This file
```

## Core Modules

### main.py

Telegram bot entry point. Handles user interactions and orchestrates the pipeline.

### downloader.py

Downloads videos from YouTube, Instagram, Twitter, etc. using `yt-dlp`.

### compiler.py

**Main video processing module**. Contains:

- `HeavyEditor` class: AI enhancement using RealESRGAN + GFPGAN
- VRAM detection and auto-configuration
- Video normalization, transitions, and final assembly
- NVENC hardware encoding support

### audio_processing.py

Audio remixing and enhancement.

### copyright_checker.py

YouTube copyright detection.

### uploader.py

YouTube video upload with metadata.

### tools-install.py

Downloads AI model weights (RealESRGAN, GFPGAN, ParseNet).

## Key Features

- **VRAM-Based Enhancement**: Automatically adjusts quality based on available GPU memory
- **Hardware Encoding**: Uses NVENC when available for faster processing
- **Graceful Fallback**: Continues processing even if AI enhancement fails
