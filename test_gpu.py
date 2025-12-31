"""
WhisperFlow Desktop - GPU Test
CUDA/GPU configuration validation script
"""

import sys
from colorama import init, Fore, Style

init()  # Initialize colorama for Windows


def print_header(text: str):
    """Displays a formatted header"""
    print(f"\n{Fore.CYAN}{'='*50}")
    print(f"  {text}")
    print(f"{'='*50}{Style.RESET_ALL}\n")


def print_success(text: str):
    """Displays a success message"""
    print(f"{Fore.GREEN}✓ {text}{Style.RESET_ALL}")


def print_error(text: str):
    """Displays an error message"""
    print(f"{Fore.RED}✗ {text}{Style.RESET_ALL}")


def print_info(text: str):
    """Displays information"""
    print(f"{Fore.YELLOW}→ {text}{Style.RESET_ALL}")


def test_pytorch():
    """Tests PyTorch installation"""
    print_header("PyTorch Test")
    
    try:
        import torch
        print_success(f"PyTorch version: {torch.__version__}")
        return True
    except ImportError as e:
        print_error(f"PyTorch not installed: {e}")
        return False


def test_cuda():
    """Tests CUDA availability"""
    print_header("CUDA Test")
    
    import torch
    
    if torch.cuda.is_available():
        print_success("CUDA is available!")
        print_info(f"CUDA version: {torch.version.cuda}")
        print_info(f"Number of GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print_info(f"GPU {i}: {props.name}")
            print_info(f"  - Total memory: {props.total_memory / 1024**3:.1f} GB")
            print_info(f"  - Compute Capability: {props.major}.{props.minor}")
        
        return True
    else:
        print_error("CUDA is not available!")
        print_info("Check that NVIDIA drivers are installed")
        print_info("Check that PyTorch CUDA is properly installed")
        return False


def test_memory_allocation():
    """Tests GPU memory allocation"""
    print_header("GPU Memory Allocation Test")
    
    import torch
    
    try:
        # Allocate a test tensor on the GPU
        test_tensor = torch.randn(1000, 1000, device="cuda")
        print_success(f"Allocation successful: tensor shape {test_tensor.shape}")
        
        # Free memory
        del test_tensor
        torch.cuda.empty_cache()
        
        # Display memory state
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        print_info(f"Allocated memory: {allocated:.1f} MB")
        print_info(f"Reserved memory: {reserved:.1f} MB")
        
        return True
    except Exception as e:
        print_error(f"Allocation error: {e}")
        return False


def test_float16_support():
    """Tests Float16 (Half Precision) support"""
    print_header("Float16 (Half Precision) Test")
    
    import torch
    
    try:
        # Create a Float16 tensor
        tensor_fp16 = torch.randn(100, 100, dtype=torch.float16, device="cuda")
        
        # Perform an operation
        result = torch.matmul(tensor_fp16, tensor_fp16.T)
        
        print_success(f"Float16 works correctly")
        print_info(f"Result shape: {result.shape}, dtype: {result.dtype}")
        
        del tensor_fp16, result
        torch.cuda.empty_cache()
        
        return True
    except Exception as e:
        print_error(f"Float16 error: {e}")
        return False


def test_transformers():
    """Tests Transformers installation"""
    print_header("Transformers Test")
    
    try:
        import transformers
        print_success(f"Transformers version: {transformers.__version__}")
        
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
        print_success("Critical imports OK")
        
        return True
    except ImportError as e:
        print_error(f"Transformers not installed: {e}")
        return False


def test_audio():
    """Tests audio dependencies"""
    print_header("Audio Test (SoundDevice)")
    
    try:
        import sounddevice as sd
        print_success(f"SoundDevice version: {sd.__version__}")
        
        # List audio devices
        devices = sd.query_devices()
        default_input = sd.query_devices(kind='input')
        
        print_info(f"Default input device: {default_input['name']}")
        print_info(f"Supported rates: {default_input['default_samplerate']} Hz")
        
        return True
    except Exception as e:
        print_error(f"Audio error: {e}")
        return False


def main():
    """Runs all tests"""
    print(f"\n{Fore.MAGENTA}{'#'*50}")
    print(f"#  WhisperFlow Desktop - GPU Diagnostic")
    print(f"{'#'*50}{Style.RESET_ALL}")
    
    results = []
    
    # Sequential tests
    results.append(("PyTorch", test_pytorch()))
    
    if results[-1][1]:  # If PyTorch OK
        results.append(("CUDA", test_cuda()))
        
        if results[-1][1]:  # If CUDA OK
            results.append(("Memory Allocation", test_memory_allocation()))
            results.append(("Float16", test_float16_support()))
    
    results.append(("Transformers", test_transformers()))
    results.append(("Audio", test_audio()))
    
    # Summary
    print_header("SUMMARY")
    
    all_passed = True
    for name, passed in results:
        if passed:
            print_success(f"{name}: OK")
        else:
            print_error(f"{name}: FAILED")
            all_passed = False
    
    print()
    if all_passed:
        print(f"{Fore.GREEN}{'='*50}")
        print("  🚀 SYSTEM READY FOR WHISPERFLOW!")
        print(f"{'='*50}{Style.RESET_ALL}")
        return 0
    else:
        print(f"{Fore.RED}{'='*50}")
        print("  ⚠️  SOME TESTS FAILED")
        print("  Fix errors before launching the application")
        print(f"{'='*50}{Style.RESET_ALL}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
