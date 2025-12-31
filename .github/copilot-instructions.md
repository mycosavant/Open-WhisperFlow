# WhisperFlow - AI Agent Instructions

## Architecture

PyQt6 local voice transcription application using Whisper Large V3 Turbo. 4-layer architecture:

```
main.py                    → Entry point, dependency verification
config.py                  → Centralized configuration (dataclasses)
src/
  audio_engine.py          → Audio capture (SoundDevice, 16kHz mono)
  transcription_service.py → Whisper pipeline (transformers, GPU/CUDA)
  ui/
    main_window.py         → Main PyQt6 window (frameless, draggable)
    workers.py             → QThread workers for async tasks
    styles.py              → Dark theme (QSS)
  utils/
    hotkey_listener.py     → Global shortcuts (pynput)
    settings.py            → JSON settings persistence
    clipboard.py           → Copy/auto-type text
```

## Data Flow

1. **Push-to-Talk (F2)** → `GlobalHotkeyListener` triggers `AudioRecorderWorker`
2. **AudioEngine** captures into numpy buffer (16kHz float32)
3. **TranscriptionWorker** sends to `TranscriptionService` (GPU)
4. Result → clipboard copy or direct typing via `type_text()`

## Project Conventions

### Configuration

- **Always use dataclasses from** [config.py](../config.py): `app_config`, `audio_config`, `model_config`, `hotkey_config`, `ui_config`
- Don't hardcode configuration values in modules

### PyQt6 Threading

- Use `QThread` with signals/slots for long tasks (see [workers.py](../src/ui/workers.py))
- Pattern: Worker with `QMutex`/`QWaitCondition` for communication
- Audio callbacks (`_audio_callback`) run in a separate thread - always copy data

### Application State

Use the `AppState` enum in [main_window.py](../src/ui/main_window.py):

- `LOADING` → model loading (~30s)
- `READY` → ready to record
- `RECORDING` → capture in progress
- `PROCESSING` → GPU transcription

### Local Imports

Modules use `sys.path.append('..')` for relative imports. Example:

```python
import sys
sys.path.append('..')
from config import audio_config
```

## Development Commands

```bash
# Installation (Windows)
setup.bat                  # Creates venv + installs PyTorch CUDA + dependencies

# Launch
python main.py             # or run.bat / WhisperFlow.bat

# GPU test
python test_gpu.py         # Checks CUDA and available memory
```

## Critical Dependencies

- **PyTorch CUDA**: installed separately via `--index-url https://download.pytorch.org/whl/cu121`
- **transformers**: `automatic-speech-recognition` pipeline with `AutoModelForSpeechSeq2Seq`
- **sounddevice**: audio stream with callbacks
- **pynput**: global key capture (works out of focus)

## Important Points

1. **GPU Memory**: Model uses ~5-6 GB VRAM. Use `torch.cuda.empty_cache()` if needed
2. **16kHz Audio**: Whisper expects 16kHz mono - conversion is automatic in `AudioEngine`
3. **Frameless Window**: Drag is handled manually in `TitleBar` via `mousePressEvent`/`mouseMoveEvent`
4. **Flash Attention**: Optional, fallback on SDPA if not installed

## User Settings Structure

`user_settings.json` file at root, managed by `SettingsManager`:

```json
{
  "push_to_talk_key": "f2",
  "output_mode": "type",
  "language": "fr"
}
```
