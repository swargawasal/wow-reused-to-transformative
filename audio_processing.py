import os
import logging
import random
import numpy as np
import librosa
import soundfile as sf
from scipy import signal
import subprocess

logger = logging.getLogger("audio_processing")

# Configuration
TRANSITION_INTERVAL = int(os.getenv("TRANSITION_INTERVAL", "10"))
FFMPEG_BIN = os.getenv("FFMPEG_BIN", "ffmpeg")

def _generate_sine_wave(freq, duration, sr=44100, vol=0.5):
    t = np.linspace(0, duration, int(sr * duration), False)
    return np.sin(freq * 2 * np.pi * t) * vol

def _create_kick_oneshot(sr=44100):
    duration = 0.3
    t = np.linspace(0, duration, int(sr * duration))
    freq = np.linspace(120, 40, len(t))
    envelope = np.exp(-12 * t)
    wave = np.sin(2 * np.pi * freq * t) * envelope
    return wave * 0.9

def _create_impact_oneshot(sr=44100):
    """Cinematic impact sound."""
    duration = 1.5
    t = np.linspace(0, duration, int(sr * duration))
    noise = np.random.uniform(-1, 1, len(t)) * np.exp(-3 * t)
    # Lowpass filter for 'boom'
    sos = signal.butter(2, 100, 'low', fs=sr, output='sos')
    boom = signal.sosfilt(sos, noise)
    return boom * 0.8

def _add_beat_accents(y, sr, interval_sec):
    """Add impact/beat every interval."""
    impact = _create_impact_oneshot(sr)
    y_out = y.copy()
    
    total_samples = y.shape[1]
    interval_samples = int(interval_sec * sr)
    
    # Start from 0 or interval? Let's start from interval to avoid boom at 0:00
    for idx in range(interval_samples, total_samples, interval_samples):
        if idx + len(impact) < total_samples:
            # Mix impact
            y_out[0, idx:idx+len(impact)] += impact * 0.4
            y_out[1, idx:idx+len(impact)] += impact * 0.4
            
    return y_out

def _micro_glitch(y, sr):
    """Randomly repeat small chunks."""
    y_out = y.copy()
    num_glitches = random.randint(2, 5)
    
    for _ in range(num_glitches):
        start = random.randint(0, y.shape[1] - 4000)
        length = random.randint(500, 2000) # 10-40ms
        chunk = y[:, start:start+length]
        
        # Repeat 2-4 times
        repeats = random.randint(2, 4)
        for r in range(repeats):
            pos = start + length * r
            if pos + length < y.shape[1]:
                y_out[:, pos:pos+length] = chunk
                
    return y_out

def process_video_audio(video_path: str) -> str:
    """
    Process audio from video:
    1. Extract
    2. Heavy Remix (Granular + Beats)
    3. Return path to new audio
    """
    logger.info(f"🎧 Processing Audio: {video_path}")
    base, _ = os.path.splitext(video_path)
    temp_wav = f"{base}_temp.wav"
    remixed_wav = f"{base}_remixed.wav"
    
    try:
        # Extract
        cmd = f'{FFMPEG_BIN} -y -i "{video_path}" -vn -acodec pcm_s16le "{temp_wav}" -v error'
        os.system(cmd)
        
        if not os.path.exists(temp_wav):
            logger.error("❌ Audio extraction failed.")
            return None
            
        # Remix
        if heavy_remix(temp_wav, remixed_wav):
            os.remove(temp_wav)
            return remixed_wav
        else:
            return temp_wav # Fallback to original extracted

    except Exception as e:
        logger.error(f"Audio processing failed: {e}")
        return None

def heavy_remix(input_path: str, output_path: str, duration: float = None) -> bool:
    """
    Professional Audio Remix - Clean & Transformative:
    1. Convert to WAV via FFmpeg (Safe Load)
    2. Load with Librosa
    3. Professional DSP Chain (No Crackling)
    4. Save
    """
    logger.info(f"🎛️ Professional Remix: {input_path}")
    temp_safe_wav = input_path + ".safe.wav"
    
    y = None
    sr = 44100

    try:
        # 1. Safe Convert to WAV (Fixes PySoundFile/MP4 issues)
        cmd = [
            FFMPEG_BIN, "-y", "-i", input_path,
            "-vn", "-ac", "2", "-ar", "44100", "-f", "wav",
            temp_safe_wav
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        
        # 2. Load
        y, sr = librosa.load(temp_safe_wav, sr=None, mono=False)
        if y.ndim == 1: y = np.stack([y, y])

    except Exception as e:
        if duration and duration > 0:
            logger.warning(f"⚠️ Audio extraction failed (likely silent video). Generating fresh audio track for {duration:.1f}s...")
            sr = 44100
            y = np.zeros((2, int(duration * sr)))
        else:
            logger.error(f"❌ Audio extraction failed: {e}")
            return False
        
        # 3. Smooth Time Stretch (±4% - subtle but noticeable)
        rate = random.uniform(0.96, 1.04)
        y_stretch = librosa.effects.time_stretch(y, rate=rate)
        logger.info(f"   🎵 Time stretch: {rate:.2f}x")
        
        # 4. Gentle Pitch Shift (±0.7 semitones - transformative but smooth)
        steps = random.uniform(-0.7, 0.7)
        y_shifted = librosa.effects.pitch_shift(y_stretch, sr=sr, n_steps=steps)
        logger.info(f"   🎵 Pitch shift: {steps:+.1f} semitones")
        
        # 5. EQ Enhancement (Professional Sound)
        from scipy.signal import butter, lfilter
        
        # Bass boost (gentle)
        b_low, a_low = butter(2, 150 / (sr / 2), btype='low')
        bass = lfilter(b_low, a_low, y_shifted)
        
        # Treble enhancement (clarity)
        b_high, a_high = butter(2, 8000 / (sr / 2), btype='high')
        treble = lfilter(b_high, a_high, y_shifted)
        
        # Mix: Original + gentle bass + subtle treble
        y_eq = y_shifted + bass * 0.15 + treble * 0.1
        logger.info(f"   🎚️ EQ applied (bass + treble)")
        
        # 6. Beat Drops (Synced to Interval) - Clean impacts
        y_beats = _add_beat_accents(y_eq, sr, TRANSITION_INTERVAL)
        
        # 7. NO GLITCHES - They cause crackling
        # Instead, add smooth fades at random points
        num_fades = random.randint(2, 4)
        y_smooth = y_beats.copy()
        for _ in range(num_fades):
            pos = random.randint(sr, y_smooth.shape[1] - sr)
            fade_len = random.randint(500, 1500)
            # Create smooth fade
            fade_curve = np.linspace(1.0, 0.7, fade_len)
            if pos + fade_len < y_smooth.shape[1]:
                y_smooth[:, pos:pos+fade_len] *= fade_curve
        logger.info(f"   ✨ Added {num_fades} smooth fades")
        
        # 8. Subtle Stereo Widening
        # Enhance stereo image without phase issues
        mid = (y_smooth[0] + y_smooth[1]) / 2
        side = (y_smooth[0] - y_smooth[1]) / 2
        # Widen by 20%
        y_wide = np.stack([mid + side * 1.2, mid - side * 1.2])
        logger.info(f"   🎧 Stereo widening applied")
        
        # 9. Gentle Reverb (5% mix - very subtle)
        reverb_length = int(sr * 0.2)  # 200ms reverb
        reverb_ir = np.random.randn(reverb_length) * np.exp(-np.arange(reverb_length) / (sr * 0.2))
        y_reverb = y_wide.copy()
        for ch in range(2):
            y_reverb[ch] = np.convolve(y_wide[ch], reverb_ir, mode='same')
        # Mix 5% reverb (very subtle)
        y_mixed = y_wide * 0.95 + y_reverb * 0.05
        logger.info(f"   🌊 Subtle reverb applied")
        
        # 10. Soft Limiter (Prevent clipping, no harsh compression)
        # Gentle limiting at 0.95 to prevent distortion
        limit = 0.95
        y_limited = y_mixed.copy()
        y_limited = np.clip(y_limited, -limit, limit)
        logger.info(f"   🛡️ Soft limiter applied")
        
        # 11. Final Normalize (-0.5dB for safety)
        max_val = np.max(np.abs(y_limited))
        if max_val > 0:
            y_final = y_limited / max_val * 0.94  # approx -0.5dB
        else:
            y_final = y_limited
            
        sf.write(output_path, y_final.T, sr)
        logger.info("✅ Professional Remix complete - Clean, smooth, transformative!")
        return True
        
    except Exception as e:
        logger.error(f"Remix failed: {e}")
        return False
    finally:
        if os.path.exists(temp_safe_wav):
            os.remove(temp_safe_wav)

