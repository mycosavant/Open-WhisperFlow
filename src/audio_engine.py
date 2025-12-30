"""
WhisperFlow Desktop - Audio Engine
Low-latency audio capture engine using SoundDevice
"""

from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Optional

import numpy as np
import sounddevice as sd

sys.path.append('..')
from config import audio_config

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass(slots=True, frozen=True)
class AudioChunk:
    """Represents a captured audio segment"""
    data: NDArray[np.float32]
    timestamp: float
    sample_rate: int
    duration: float


class AudioEngine:
    """
    Audio capture engine for Push-to-Talk
    
    Features:
    - Real-time capture at 16kHz (optimal for Whisper)
    - Circular buffer to avoid data loss
    - Push-to-Talk mode with start/stop
    - Automatic signal normalization
    """
    
    def __init__(
        self,
        sample_rate: int = audio_config.SAMPLE_RATE,
        channels: int = audio_config.CHANNELS,
        blocksize: int = audio_config.BLOCKSIZE,
        device: Optional[int] = None
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.blocksize = blocksize
        self.device = device
        
        # Recording state
        self._is_recording = False
        self._recording_lock = threading.Lock()
        
        # Buffer for audio data
        self._audio_buffer: list[np.ndarray] = []
        self._buffer_lock = threading.Lock()
        
        # Audio stream
        self._stream: Optional[sd.InputStream] = None
        
        # Callbacks
        self._on_audio_level: Optional[Callable[[float], None]] = None
        
        # Statistics
        self._start_time: float = 0
        self._chunks_captured: int = 0
    
    def set_audio_level_callback(self, callback: Callable[[float], None]):
        """Sets the callback for audio levels (VU meter)"""
        self._on_audio_level = callback
    
    def _audio_callback(
        self,
        indata: NDArray[np.float32],
        frames: int,
        time_info,
        status: sd.CallbackFlags
    ) -> None:
        """
        Callback called by SoundDevice for each audio block.
        WARNING: This callback runs in a separate thread!
        """
        if status:
            print(f"⚠️ Audio status: {status}")
        
        # Quick check without lock if possible
        if not self._is_recording:
            return
        
        with self._recording_lock:
            if not self._is_recording:
                return
        
        # Copy data (important because indata is a reused buffer)
        # Use ravel() if already contiguous, otherwise flatten()
        audio_data: NDArray[np.float32] = indata.ravel().copy() if indata.flags['C_CONTIGUOUS'] else indata.flatten()
        
        # Optimized audio level calculation (RMS) with np.dot
        callback = self._on_audio_level
        if callback is not None:
            rms = np.sqrt(np.dot(audio_data, audio_data) / len(audio_data))
            # Use np.clip and optimized log calculation
            level_db = 20.0 * np.log10(max(rms, 1e-10))
            normalized_level = np.clip((level_db + 60.0) / 60.0, 0.0, 1.0)
            callback(float(normalized_level))
        
        # Add to buffer
        with self._buffer_lock:
            self._audio_buffer.append(audio_data)
            self._chunks_captured += 1
    
    def start_recording(self) -> bool:
        """
        Starts audio recording
        Returns: True if started successfully
        """
        with self._recording_lock:
            if self._is_recording:
                return False
            
            # Reset buffer
            with self._buffer_lock:
                self._audio_buffer.clear()
                self._chunks_captured = 0
            
            self._start_time = time.time()
            
            try:
                # Create and start stream
                self._stream = sd.InputStream(
                    samplerate=self.sample_rate,
                    channels=self.channels,
                    dtype=audio_config.DTYPE,
                    blocksize=self.blocksize,
                    device=self.device,
                    callback=self._audio_callback
                )
                self._stream.start()
                self._is_recording = True
                return True
                
            except Exception as e:
                print(f"❌ Audio startup error: {e}")
                return False
    
    def stop_recording(self) -> Optional[AudioChunk]:
        """
        Stops recording and returns captured data
        Returns: AudioChunk containing recorded audio, or None
        """
        with self._recording_lock:
            if not self._is_recording:
                return None
            
            self._is_recording = False
        
        # Stop stream
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        
        # Retrieve and concatenate data
        with self._buffer_lock:
            if not self._audio_buffer:
                return None
            
            # Pre-calculate total size to avoid reallocations
            total_samples = sum(chunk.size for chunk in self._audio_buffer)
            audio_data = np.empty(total_samples, dtype=np.float32)
            
            # Efficient copy into pre-allocated buffer
            offset = 0
            for chunk in self._audio_buffer:
                audio_data[offset:offset + chunk.size] = chunk
                offset += chunk.size
            
            self._audio_buffer.clear()
        
        # Normalize audio (-1 to 1) - in-place operation
        max_val = np.abs(audio_data).max()
        if max_val > 0:
            audio_data *= (0.95 / max_val)
        
        duration = len(audio_data) / self.sample_rate
        
        return AudioChunk(
            data=audio_data,
            timestamp=self._start_time,
            sample_rate=self.sample_rate,
            duration=duration
        )
    
    @property
    def is_recording(self) -> bool:
        """Returns True if recording is in progress"""
        with self._recording_lock:
            return self._is_recording
    
    @staticmethod
    def list_devices() -> list[dict]:
        """Lists all available audio devices"""
        devices = sd.query_devices()
        input_devices = []
        
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                input_devices.append({
                    'id': i,
                    'name': device['name'],
                    'channels': device['max_input_channels'],
                    'sample_rate': device['default_samplerate'],
                    'is_default': device == sd.query_devices(kind='input')
                })
        
        return input_devices
    
    @staticmethod
    def get_default_device() -> dict:
        """Returns the default input device"""
        device = sd.query_devices(kind='input')
        return {
            'name': device['name'],
            'channels': device['max_input_channels'],
            'sample_rate': device['default_samplerate']
        }


class AudioLevelMonitor:
    """
    Audio level monitor for visual feedback
    Provides a smoothed audio level value
    """
    
    def __init__(self, smoothing: float = 0.3):
        self.smoothing = smoothing
        self._current_level = 0.0
        self._peak_level = 0.0
        self._peak_decay = 0.95
        self._lock = threading.Lock()
    
    def update(self, level: float):
        """Updates level with smoothing"""
        with self._lock:
            # Exponential smoothing
            self._current_level = (
                self.smoothing * level + 
                (1 - self.smoothing) * self._current_level
            )
            
            # Peak hold with decay
            if level > self._peak_level:
                self._peak_level = level
            else:
                self._peak_level *= self._peak_decay
    
    @property
    def level(self) -> float:
        """Current level (0-1)"""
        with self._lock:
            return self._current_level
    
    @property
    def peak(self) -> float:
        """Peak level (0-1)"""
        with self._lock:
            return self._peak_level


# Standalone test
if __name__ == "__main__":
    print("🎤 Audio engine test")
    print("-" * 40)
    
    # List devices
    print("\n📋 Available input devices:")
    for device in AudioEngine.list_devices():
        marker = "→ " if device['is_default'] else "  "
        print(f"{marker}[{device['id']}] {device['name']}")
    
    # Recording test
    print("\n🔴 Recording 3 seconds...")
    
    engine = AudioEngine()
    monitor = AudioLevelMonitor()
    
    engine.set_audio_level_callback(monitor.update)
    
    engine.start_recording()
    
    for i in range(30):
        time.sleep(0.1)
        bar = "█" * int(monitor.level * 20)
        print(f"\r  Level: [{bar:<20}] {monitor.level:.2f}", end="")
    
    print()
    
    chunk = engine.stop_recording()
    
    if chunk:
        print(f"\n✅ Recording completed!")
        print(f"   Duration: {chunk.duration:.2f}s")
        print(f"   Samples: {len(chunk.data)}")
        print(f"   Sample Rate: {chunk.sample_rate} Hz")
    else:
        print("\n❌ No data captured")
