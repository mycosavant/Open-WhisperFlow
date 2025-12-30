"""
WhisperFlow Desktop
===================

Real-time local voice transcription application.
Uses Whisper Large V3 Turbo optimized for GPU (RTX 4080).

Usage:
    python main.py

Shortcuts:
    F2      - Push-to-Talk (hold to record)
    F3      - Copy transcription
    ESC     - Quit

Author: WhisperFlow Team
License: MIT
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

from config import app_config


def check_requirements():
    """Verify all dependencies are installed"""
    missing = []
    
    try:
        import PyQt6
    except ImportError:
        missing.append("PyQt6")
    
    try:
        import torch
        if not torch.cuda.is_available():
            print("⚠️  WARNING: CUDA is not available!")
            print("   Transcription will be very slow without GPU.")
            print()
    except ImportError:
        missing.append("torch")
    
    try:
        import faster_whisper
    except ImportError:
        missing.append("faster-whisper")
    
    try:
        import sounddevice
    except ImportError:
        missing.append("sounddevice")
    
    try:
        import pynput
    except ImportError:
        missing.append("pynput")
    
    try:
        import pyperclip
    except ImportError:
        missing.append("pyperclip")
    
    if missing:
        print("❌ Missing dependencies:")
        for dep in missing:
            print(f"   - {dep}")
        print()
        print("Install them with:")
        print("   pip install -r requirements.txt")
        print()
        print("For PyTorch with CUDA:")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        sys.exit(1)


def main():
    """Main entry point"""
    
    # Banner
    print()
    print("╔════════════════════════════════════════════════════════════╗")
    print("║                                                            ║")
    print("║   🎤 WhisperFlow Desktop                                   ║")
    print("║   Real-time local voice transcription                      ║")
    print("║                                                            ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print()
    
    # Check dependencies
    check_requirements()
    
    # High DPI configuration
    if hasattr(Qt.ApplicationAttribute, 'AA_EnableHighDpiScaling'):
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling, True)
    if hasattr(Qt.ApplicationAttribute, 'AA_UseHighDpiPixmaps'):
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)
    
    # Create application
    app = QApplication(sys.argv)
    app.setApplicationName(app_config.APP_NAME)
    app.setApplicationVersion(app_config.APP_VERSION)
    
    # Default font
    font = QFont("Segoe UI", 10)
    app.setFont(font)
    
    # Display GPU info
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"🖥️  GPU detected: {gpu_name} ({gpu_mem:.1f} GB)")
        else:
            print("⚠️  CPU mode (slow)")
    except ImportError:
        print("⚠️  PyTorch not installed")
    except Exception as e:
        print(f"⚠️  GPU detection error: {e}")
    
    print()
    print("📌 Keyboard shortcuts:")
    print("   F2  - Push-to-Talk (hold to speak)")
    print("   F3  - Copy transcription")
    print("   ESC - Quit")
    print()
    print("🚀 Starting application...")
    print()
    
    # Import and create main window
    from src.ui.main_window import MainWindow
    
    window = MainWindow()
    window.show()
    
    # Event loop
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
