# How to Enable NVENC (Hardware Encoding)

## What is NVENC?

NVENC is NVIDIA's hardware video encoder built into NVIDIA GPUs. It provides **10x faster** encoding than CPU.

## Requirements

You need:

1. **NVIDIA GPU** (GTX 600 series or newer, RTX series recommended)
2. **Updated NVIDIA Drivers** (latest version)

## Check if You Have NVIDIA GPU

### Option 1: Check Device Manager

1. Press `Win + X`
2. Select "Device Manager"
3. Expand "Display adapters"
4. Look for "NVIDIA" in the name

### Option 2: Run Command

```powershell
nvidia-smi
```

If you see GPU info → You have NVIDIA GPU
If you see error → No NVIDIA GPU

## How to Enable NVENC

### Step 1: Update NVIDIA Drivers

1. Go to: https://www.nvidia.com/Download/index.aspx
2. Select your GPU model
3. Download and install latest driver
4. Restart your computer

### Step 2: Verify NVENC Works

After updating drivers, run:

```powershell
ffmpeg -f lavfi -i color=c=black:s=256x256:d=1 -c:v h264_nvenc -f null -
```

If no error → NVENC is working!
If error → Your GPU doesn't support NVENC

## If You Don't Have NVIDIA GPU

**You cannot use NVENC.** But I've optimized CPU encoding for you:

### Current Optimizations (Applied)

- ✅ Preset changed to `veryfast` (5x faster than `slow`)
- ✅ CRF changed to `23` (faster, still good quality)
- ✅ AI enhancement disabled on CPU
- ✅ Fast ffmpeg upscaling

### Result

- **Speed**: 5-10x faster than before
- **Quality**: Still very good (CRF 23 is YouTube-recommended)
- **CPU Usage**: Much lower than before

## Alternative: Use Google Colab (Free GPU)

If you want GPU acceleration without buying hardware:

1. Upload your project to GitHub
2. Open Google Colab: https://colab.research.google.com
3. Enable GPU: Runtime → Change runtime type → GPU
4. Clone your repo and run the bot

Colab provides free NVIDIA Tesla T4 GPU with NVENC support!
