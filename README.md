# WhisperFlow Desktop

🎤 **Real-time local voice transcription application**

Transform your voice to text instantly, privately, without cloud connection.

![WhisperFlow](https://img.shields.io/badge/WhisperFlow-v1.0.0-blue)
![Python](https://img.shields.io/badge/Python-3.10+-green)
![CUDA](https://img.shields.io/badge/CUDA-12.1-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## ✨ Features

- 🚀 **Ultra-fast** - Real-time transcription with GPU acceleration
- 🔒 **100% Local** - No data leaves your computer
- 🎯 **Accurate** - Uses OpenAI's Whisper Large V3 Turbo
- 🎹 **Push-to-Talk** - Press F2, speak, release, it's transcribed
- 📋 **Easy Copy** - Result copied with one click or F3
- 🎨 **Modern UI** - Minimalist floating interface macOS style

---

## 🖥️ Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **GPU** | NVIDIA GTX 1060 (6GB) | RTX 3080+ / RTX 4080 |
| **VRAM** | 6 GB | 12+ GB |
| **RAM** | 8 GB | 16+ GB |
| **OS** | Windows 10 | Windows 11 |
| **Python** | 3.10 | 3.11 |

### Required Software

1. **Python 3.10+** - [Download](https://python.org)
2. **Recent NVIDIA Drivers** - [Download](https://nvidia.com/drivers)
3. **FFmpeg** (optional) - [Download](https://ffmpeg.org)

---

## 🚀 Installation

### Automatic Installation (recommended)

```bash
# 1. Clone or download the project
cd WhisperFlow

# 2. Run the installation
setup.bat
```

The `setup.bat` script will automatically:
- Create a Python virtual environment
- Install PyTorch with CUDA 12.1 support
- Install all dependencies
- Test GPU configuration
- Launch the application

### Manual Installation

```bash
# 1. Create virtual environment
python -m venv .venv
.venv\Scripts\activate

# 2. Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. Install dependencies
pip install -r requirements.txt

# 4. Test GPU
python test_gpu.py

# 5. Launch application
python main.py
```

---

## 🎮 Usage

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| **F2** | Push-to-Talk (hold to speak) |
| **F3** | Copy transcription |
| **ESC** | Quit application |

### Typical Workflow

1. **Launch** the application with `run.bat`
2. **Wait** for model loading (~30s on first launch)
3. **Hold F2** and speak into your microphone
4. **Release F2** - transcription appears instantly
5. **Press F3** to copy or click "Copy"

---

## ⚙️ Configuration

Modify `config.py` to customize:

```python
# Transcription language
LANGUAGE = "fr"  # fr, en, es, de, etc.

# Push-to-Talk key
PUSH_TO_TALK_KEY = "f2"

# Whisper model
MODEL_ID = "openai/whisper-large-v3-turbo"
```

### Available Models

| Model | VRAM | Accuracy | Speed |
|-------|------|----------|-------|
| `whisper-tiny` | ~1 GB | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| `whisper-base` | ~1 GB | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| `whisper-small` | ~2 GB | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| `whisper-medium` | ~5 GB | ⭐⭐⭐⭐ | ⭐⭐ |
| `whisper-large-v3` | ~10 GB | ⭐⭐⭐⭐⭐ | ⭐ |
| **`whisper-large-v3-turbo`** | ~6 GB | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |

---

## 🏗️ Architecture

```
WhisperFlow/
├── main.py                 # Entry point
├── config.py               # Centralized configuration
├── requirements.txt        # Python dependencies
├── setup.bat               # Installation script
├── run.bat                 # Quick launcher
├── test_gpu.py             # GPU diagnostics
├── LICENSE                 # MIT License
└── src/
    ├── audio_engine.py           # Audio capture (SoundDevice)
    ├── transcription_service.py  # AI engine (Faster-Whisper)
    ├── smart_formatter.py        # Smart text formatting
    ├── ui/
    │   ├── main_window.py        # PyQt6 window
    │   ├── key_capture_dialog.py # Shortcut configuration
    │   ├── styles.py             # CSS styles
    │   └── workers.py            # QThread threading
    └── utils/
        ├── clipboard.py          # Clipboard & auto-type
        ├── history.py            # Transcription history
        ├── hotkey_listener.py    # Global shortcuts
        └── settings.py           # Settings persistence
```

---

## 🐛 Troubleshooting

### "CUDA is not available"

1. Verify you have an NVIDIA card
2. Update your drivers: [nvidia.com/drivers](https://nvidia.com/drivers)
3. Reinstall PyTorch: `pip install torch --index-url https://download.pytorch.org/whl/cu121`

### "Out of Memory" (insufficient VRAM)

1. Close other applications using the GPU
2. Use a smaller model in `config.py`:
   ```python
   MODEL_ID = "openai/whisper-small"
   ```

### Microphone not working

1. Verify the microphone is allowed in Windows
2. Test with `python -c "import sounddevice; print(sounddevice.query_devices())"`
3. Manually select the device in `config.py`

### Application won't start

1. Run `python test_gpu.py` to diagnose
2. Check logs in the terminal
3. Reinstall with `setup.bat`

---

## 📊 Performance

Tested on RTX 4080 (16 GB VRAM):

| Audio duration | Transcription time | RTF* |
|----------------|-------------------|------|
| 5 seconds | ~0.5s | 0.1x |
| 30 seconds | ~2s | 0.07x |
| 1 minute | ~3s | 0.05x |

*RTF (Real-Time Factor): < 1 = faster than real-time

---

## 🤝 Contributing

Contributions are welcome! Feel free to:
- 🐛 Report bugs
- 💡 Propose features
- 🔧 Submit pull requests

---

## 📄 License

MIT License - Free for personal and commercial use.

---

## 🙏 Credits

- [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper) - Optimized transcription engine
- [OpenAI Whisper](https://github.com/openai/whisper) - Transcription model
- [Hugging Face Transformers](https://huggingface.co/transformers) - ML pipeline
- [PyQt6](https://riverbankcomputing.com/software/pyqt) - GUI framework
- [pynput](https://github.com/moses-palmer/pynput) - Keyboard shortcuts
- [SoundDevice](https://python-sounddevice.readthedocs.io) - Audio capture

---

<div align="center">

**WhisperFlow Desktop** - Made with ❤️ for productivity

</div>
