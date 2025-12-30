"""
WhisperFlow Desktop - Transcription Service
Transcription service using Faster-Whisper (CTranslate2)

Faster-Whisper provides:
- ~4x faster than transformers
- ~3x less RAM/VRAM usage
- Same transcription quality
"""

from __future__ import annotations

import gc
import os
import re
import sys
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import numpy as np

sys.path.append('..')
from config import model_config, app_config

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Conditional faster-whisper import
try:
    from faster_whisper import WhisperModel
    _HAS_FASTER_WHISPER = True
except ImportError:
    _HAS_FASTER_WHISPER = False
    WhisperModel = None

# Import torch for GPU info (optional)
try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


@dataclass(slots=True, frozen=True)
class TranscriptionResult:
    """Transcription result"""
    text: str
    language: str  # Language used or detected
    duration: float  # Audio duration
    processing_time: float  # Processing time
    detected_language: str | None = None  # Detected language if auto
    confidence: float | None = None


# Pre-compiled hallucination patterns for performance
_HALLUCINATION_PATTERN = re.compile(
    r"(Merci d'avoir regardé|Sous-titres réalisés|Sous-titres par|"
    r"Merci à tous|À bientôt|Abonnez-vous|N'oubliez pas de|"
    r"Cliquez sur|\[Musique\]|\[Applaudissements\]|\(Musique\)|\.{3,})",
    re.IGNORECASE
)
_WHITESPACE_PATTERN = re.compile(r'\s+')


class TranscriptionService:
    """
    GPU-optimized Faster-Whisper transcription service
    
    Features:
    - Uses CTranslate2 for optimal performance
    - Float16/INT8 support to reduce memory
    - Integrated VAD to ignore silence
    - Multi-language support with auto-detection
    """
    
    # Model name mapping
    MODEL_MAPPING = {
        "openai/whisper-large-v3-turbo": "turbo",
        "openai/whisper-large-v3": "large-v3",
        "openai/whisper-large-v2": "large-v2",
        "openai/whisper-medium": "medium",
        "openai/whisper-small": "small",
        "openai/whisper-base": "base",
        "openai/whisper-tiny": "tiny",
    }
    
    def __init__(
        self,
        model_id: str = model_config.MODEL_ID,
        device: str = model_config.DEVICE,
        compute_type: str = "float16"
    ):
        # Convert HuggingFace model_id to faster-whisper if needed
        self.model_id = self.MODEL_MAPPING.get(model_id, model_id)
        self.device = device
        self.compute_type = compute_type
        
        # Model components
        self._model: WhisperModel | None = None
        
        # State
        self._is_loaded = False
        self._is_loading = False
        self._load_lock = threading.Lock()
        
        # Callbacks
        self._on_progress: Callable[[str, float], None] | None = None
        
        # Statistics
        self._total_transcriptions = 0
        self._total_audio_duration = 0.0
        self._total_processing_time = 0.0
    
    def set_progress_callback(self, callback: Callable[[str, float], None]) -> None:
        """Sets progress callback (message, percentage)"""
        self._on_progress = callback
    
    def _report_progress(self, message: str, progress: float):
        """Reports progress"""
        if self._on_progress:
            self._on_progress(message, progress)
    
    def load_model(self) -> bool:
        """
        Loads the Whisper model with Faster-Whisper.
        
        Returns: True if loaded successfully
        """
        if not _HAS_FASTER_WHISPER:
            print("❌ faster-whisper is not installed!")
            print("   Install with: pip install faster-whisper")
            return False
        
        with self._load_lock:
            if self._is_loaded:
                return True
            
            if self._is_loading:
                return False
            
            self._is_loading = True
        
        try:
            self._report_progress("Checking GPU...", 0.1)
            
            # Check CUDA if device=cuda
            if self.device == "cuda":
                if _HAS_TORCH and not torch.cuda.is_available():
                    print("⚠️ CUDA not available, using CPU")
                    self.device = "cpu"
                    self.compute_type = "int8"
            
            # Configure local cache
            cache_dir = str(app_config.MODELS_DIR)
            os.environ["HF_HOME"] = cache_dir
            print(f"📁 Model cache: {cache_dir}")
            
            # Free existing GPU memory
            if _HAS_TORCH and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self._report_progress("Loading Faster-Whisper model...", 0.3)
            print(f"📦 Loading '{self.model_id}' on {self.device} ({self.compute_type})...")
            
            # Load model with Faster-Whisper
            self._model = WhisperModel(
                self.model_id,
                device=self.device,
                compute_type=self.compute_type,
                download_root=cache_dir,
            )
            
            self._report_progress("Ready!", 1.0)
            
            with self._load_lock:
                self._is_loaded = True
                self._is_loading = False
            
            # Display memory usage
            if _HAS_TORCH and torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated() / 1024**3
                print(f"✅ Model loaded! VRAM used: {memory_used:.2f} GB")
            else:
                print("✅ Model loaded!")
            
            return True
            
        except Exception as e:
            print(f"❌ Model loading error: {e}")
            import traceback
            traceback.print_exc()
            with self._load_lock:
                self._is_loading = False
            return False
    
    def unload_model(self):
        """Unloads model and frees GPU memory"""
        with self._load_lock:
            if not self._is_loaded:
                return
            
            if self._model:
                del self._model
                self._model = None
            
            # Force le garbage collection
            gc.collect()
            
            if _HAS_TORCH and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self._is_loaded = False
            print("🗑️ Modèle déchargé, mémoire libérée")
    
    def transcribe(
        self,
        audio_data: NDArray[np.float32],
        sample_rate: int = 16000,
        language: str | None = None
    ) -> TranscriptionResult | None:
        """
        Transcribes an audio segment with Faster-Whisper.
        
        Args:
            audio_data: Numpy array of audio (float32, -1 to 1)
            sample_rate: Sampling rate
            language: Language code (fr, en, etc.), "auto" for detection, None uses config
            
        Returns:
            TranscriptionResult or None on error
        """
        if not self._is_loaded or self._model is None:
            print("⚠️ Model not loaded!")
            return None
        
        start_time = time.time()
        audio_duration = len(audio_data) / sample_rate
        
        # Determine language to use
        use_language = language if language is not None else model_config.LANGUAGE
        auto_detect = use_language == "auto" or use_language is None or use_language == ""
        
        try:
            # Transcription parameters
            transcribe_kwargs = {
                "beam_size": 5,
                "best_of": 1,
                "vad_filter": True,  # Automatically filters silence
                "vad_parameters": {
                    "threshold": 0.5,
                    "min_speech_duration_ms": 250,
                    "min_silence_duration_ms": 500,
                },
                "without_timestamps": True,  # Faster without timestamps
            }
            
            # If not auto-detection, specify language
            if not auto_detect:
                transcribe_kwargs["language"] = use_language
            
            # Execute transcription
            segments, info = self._model.transcribe(audio_data, **transcribe_kwargs)
            
            # Collect all segments into a single string
            text_parts = []
            for segment in segments:
                text_parts.append(segment.text.strip())
            
            text = " ".join(text_parts).strip()
            
            processing_time = time.time() - start_time
            
            # Update statistics
            self._total_transcriptions += 1
            self._total_audio_duration += audio_duration
            self._total_processing_time += processing_time
            
            # Remove common hallucinations
            text = self._clean_hallucinations(text)
            
            # Determine detected/used language
            detected_lang = info.language if auto_detect else None
            final_language = info.language if auto_detect else use_language
            confidence = info.language_probability if auto_detect else None
            
            return TranscriptionResult(
                text=text,
                language=final_language,
                duration=audio_duration,
                processing_time=processing_time,
                detected_language=detected_lang,
                confidence=confidence,
            )
            
        except Exception as e:
            print(f"❌ Transcription error: {e}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            # Force garbage collection periodically
            if self._total_transcriptions % 10 == 0:
                gc.collect()
                if _HAS_TORCH and torch.cuda.is_available():
                    torch.cuda.empty_cache()
    
    def _clean_hallucinations(self, text: str) -> str:
        """
        Removes common Whisper hallucinations.
        Uses pre-compiled regex for performance.
        """
        # Remove hallucinations with pre-compiled regex
        text = _HALLUCINATION_PATTERN.sub('', text)
        
        # Clean multiple spaces with pre-compiled regex
        text = _WHITESPACE_PATTERN.sub(' ', text)
        
        return text.strip()
    
    @property
    def is_loaded(self) -> bool:
        """Returns True if model is loaded"""
        with self._load_lock:
            return self._is_loaded
    
    @property
    def is_loading(self) -> bool:
        """Returns True if model is loading"""
        with self._load_lock:
            return self._is_loading
    
    @property
    def stats(self) -> dict:
        """Returns transcription statistics"""
        avg_rtf = 0  # Real-Time Factor
        if self._total_audio_duration > 0:
            avg_rtf = self._total_processing_time / self._total_audio_duration
        
        return {
            "total_transcriptions": self._total_transcriptions,
            "total_audio_duration": self._total_audio_duration,
            "total_processing_time": self._total_processing_time,
            "average_rtf": avg_rtf,  # < 1 = faster than real-time
        }
    
    @staticmethod
    def get_gpu_info() -> dict:
        """Returns GPU information"""
        if not _HAS_TORCH or not torch.cuda.is_available():
            return {"available": False}
        
        props = torch.cuda.get_device_properties(0)
        return {
            "available": True,
            "name": props.name,
            "total_memory_gb": props.total_memory / 1024**3,
            "memory_allocated_gb": torch.cuda.memory_allocated() / 1024**3,
            "memory_reserved_gb": torch.cuda.memory_reserved() / 1024**3,
        }
    
    @staticmethod
    def get_vram_usage() -> tuple[float, float, float]:
        """
        Returns current VRAM usage.
        
        Returns:
            Tuple (used_gb, total_gb, percentage)
        """
        if not _HAS_TORCH or not torch.cuda.is_available():
            return (0.0, 0.0, 0.0)
        
        allocated = torch.cuda.memory_allocated() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        percentage = (allocated / total * 100) if total > 0 else 0.0
        
        return (allocated, total, percentage)


# Standalone test
if __name__ == "__main__":
    print("🤖 Transcription service test (Faster-Whisper)")
    print("-" * 50)
    
    if not _HAS_FASTER_WHISPER:
        print("❌ faster-whisper is not installed!")
        print("   pip install faster-whisper")
        exit(1)
    
    # Display GPU info
    gpu_info = TranscriptionService.get_gpu_info()
    if gpu_info["available"]:
        print(f"✅ GPU: {gpu_info['name']}")
        print(f"   Total memory: {gpu_info['total_memory_gb']:.1f} GB")
    else:
        print("ℹ️ No GPU, using CPU")
    
    # Create service
    service = TranscriptionService()
    
    def on_progress(msg, progress):
        bar = "█" * int(progress * 20)
        print(f"\r  [{bar:<20}] {progress*100:.0f}% - {msg}", end="")
    
    service.set_progress_callback(on_progress)
    
    print("\n\n📦 Loading model...")
    if not service.load_model():
        print("\n❌ Loading failed!")
        exit(1)
    
    print("\n")
    
    # Test with synthetic audio (silence)
    print("🎤 Transcription test (1s silence)...")
    test_audio = np.zeros(16000, dtype=np.float32)
    result = service.transcribe(test_audio)
    
    if result:
        print(f"✅ Transcription successful!")
        print(f"   Text: '{result.text}'")
        print(f"   Language: {result.language}")
        print(f"   Time: {result.processing_time:.2f}s")
    
    # Statistics
    print(f"\n📊 After transcription:")
    gpu_info = TranscriptionService.get_gpu_info()
    if gpu_info["available"]:
        print(f"   VRAM used: {gpu_info['memory_allocated_gb']:.2f} GB")
    
    stats = service.stats
    print(f"   Average RTF: {stats['average_rtf']:.3f}")
    
    # Unload
    service.unload_model()
