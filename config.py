"""
WhisperFlow Desktop - Configuration
Centralized application configuration
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class AppConfig:
    """General application configuration"""
    APP_NAME: str = "WhisperFlow"
    APP_VERSION: str = "1.0.0"
    
    # Application directory
    BASE_DIR: Path = Path(__file__).parent
    CACHE_DIR: Path = BASE_DIR / ".cache"
    MODELS_DIR: Path = CACHE_DIR / "models"  # Hugging Face models cache
    

@dataclass
class AudioConfig:
    """Audio engine configuration"""
    SAMPLE_RATE: int = 16000  # Whisper expects 16kHz
    CHANNELS: int = 1  # Mono
    DTYPE: str = "float32"
    BLOCKSIZE: int = 1024  # Audio buffer size
    

@dataclass
class ModelConfig:
    """Whisper model configuration"""
    MODEL_ID: str = "turbo"  # Faster-Whisper model (turbo = large-v3-turbo)
    DEVICE: str = "cuda"  # GPU usage
    TORCH_DTYPE: str = "float16"  # Half precision to save VRAM
    
    # Generation parameters
    LANGUAGE: str = "fr"  # Default language ("auto" for automatic detection)
    TASK: str = "transcribe"
    MAX_NEW_TOKENS: int = 440  # Reduced to leave room for startup tokens (4)
    
    # Optimizations
    USE_FLASH_ATTENTION: bool = True
    CHUNK_LENGTH_S: int = 30  # Max segment duration
    
    # Batch mode for long transcriptions
    BATCH_SIZE: int = 8  # Number of chunks to process in parallel
    LONG_AUDIO_THRESHOLD_S: float = 30.0  # Threshold to activate batch mode


@dataclass
class HotkeyConfig:
    """Keyboard shortcuts configuration"""
    PUSH_TO_TALK_KEY: str = "f2"
    COPY_TO_CLIPBOARD_KEY: str = "f3"
    QUIT_KEY: str = "escape"
    
    # Output mode: "type" = direct typing, "clipboard" = copy only
    OUTPUT_MODE: str = "type"  # Auto-types in active app
    TYPE_DELAY: float = 0.01  # Delay between characters (seconds)


@dataclass
class UIConfig:
    """User interface configuration"""
    WINDOW_WIDTH: int = 440
    WINDOW_HEIGHT: int = 260
    WINDOW_OPACITY: float = 0.95
    
    # VRAM update interval (ms)
    VRAM_UPDATE_INTERVAL_MS: int = 2000
    
    # Colors
    COLOR_BACKGROUND: str = "#1E1E2E"
    COLOR_SURFACE: str = "#313244"
    COLOR_TEXT: str = "#CDD6F4"
    COLOR_TEXT_DIM: str = "#6C7086"
    COLOR_ACCENT: str = "#89B4FA"
    COLOR_RECORDING: str = "#F38BA8"
    COLOR_PROCESSING: str = "#89DCEB"
    COLOR_SUCCESS: str = "#A6E3A1"
    
    # Borders and shadows
    BORDER_RADIUS: int = 16
    SHADOW_BLUR: int = 20


# Configuration instances
app_config = AppConfig()
audio_config = AudioConfig()
model_config = ModelConfig()
hotkey_config = HotkeyConfig()
ui_config = UIConfig()

# Create cache directories if needed
app_config.CACHE_DIR.mkdir(exist_ok=True)
app_config.MODELS_DIR.mkdir(exist_ok=True)
