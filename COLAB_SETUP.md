# Google Colab Setup Guide

## Running the Bot in Google Colab

### ✅ Compatibility Status

This bot is **fully compatible** with both:

- 🖥️ **Local Windows** (CPU mode)
- ☁️ **Google Colab** (GPU mode with AI enhancement)

### Setup Instructions for Colab

#### 1. Enable GPU Runtime

```
Runtime → Change runtime type → Hardware accelerator → GPU → Save
```

#### 2. Install System Dependencies

```python
# Install FFmpeg
!apt-get update
!apt-get install -y ffmpeg

# Verify installation
!ffmpeg -version
```

#### 3. Clone Your Repository

```python
# Clone from GitHub
!git clone https://github.com/yourusername/youtube-automation.git
%cd youtube-automation/yt
```

#### 4. Install Python Dependencies

```python
!pip install -r requirements.txt
```

#### 5. Setup Environment Variables

```python
# Create .env file
import os
from google.colab import userdata

# Get secrets from Colab Secrets
telegram_token = userdata.get('TELEGRAM_BOT_TOKEN')
ig_username = userdata.get('IG_USERNAME')
ig_password = userdata.get('IG_PASSWORD')

# Write to .env
with open('.env', 'w') as f:
    f.write(f"TELEGRAM_BOT_TOKEN={telegram_token}\n")
    f.write(f"IG_USERNAME={ig_username}\n")
    f.write(f"IG_PASSWORD={ig_password}\n")
    f.write("COMPUTE_MODE=gpu\n")
    f.write("ENHANCEMENT_LEVEL=2x\n")
    f.write("FORCE_AUDIO_REMIX=yes\n")
```

#### 6. Run the Bot

```python
!python main.py
```

### Key Differences: Local vs Colab

| Feature          | Local (CPU) | Colab (GPU)        |
| ---------------- | ----------- | ------------------ |
| AI Enhancement   | ❌ Disabled | ✅ Enabled         |
| Processing Speed | Moderate    | Fast               |
| Video Quality    | Good        | Excellent          |
| Cost             | Free        | Free (with limits) |
| Session Duration | Unlimited   | 12 hours max       |

### GPU Detection

The bot automatically detects GPU:

- **Colab**: Tesla T4/K80/P100 GPU detected → AI enhancement ON
- **Local**: No GPU → AI enhancement OFF, uses FFmpeg

### Colab Limitations

- ⏰ **Session timeout**: 12 hours maximum
- 💾 **Storage**: Temporary (files deleted after session)
- 🔄 **Reconnection**: May disconnect after inactivity

### Recommended Workflow

1. **Development**: Test locally (CPU mode)
2. **Production**: Run in Colab (GPU mode) for best quality
3. **Automation**: Use Colab scheduled notebooks

### Troubleshooting

**Issue**: Bot can't access Telegram
**Solution**: Colab blocks some network connections, use ngrok tunnel

**Issue**: Files disappear after session
**Solution**: Mount Google Drive to save outputs

```python
from google.colab import drive
drive.mount('/content/drive')
```

**Issue**: GPU not detected
**Solution**: Verify runtime type is set to GPU, restart runtime

### Pro Tips

- 💡 Save processed videos to Google Drive
- 💡 Use Colab Pro for longer sessions
- 💡 Monitor GPU usage with `!nvidia-smi`
- 💡 Keep browser tab active to prevent disconnection
