import torch
import subprocess

def log_memory_usage():
    """Comprehensive GPU memory logging that shows memory usage"""
    if not torch.cuda.is_available():
        # print("CUDA not available")
        return
    
    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    max_allocated = torch.cuda.max_memory_allocated() / 1024**2
    
    # Total GPU memory
    device = torch.cuda.current_device()
    total_memory = torch.cuda.get_device_properties(device).total_memory / 1024**2
    
    print(f"=== PyTorch Memory View ===")
    print(f"Allocated: {allocated:.2f} MB")
    print(f"Reserved: {reserved:.2f} MB") 
    print(f"Peak: {max_allocated:.2f} MB")
    print(f"Free (PyTorch view): {total_memory - reserved:.2f} MB")
    
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.free,memory.total', 
                               '--format=csv,noheader,nounits'], 
                               capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            used, free, total = map(int, result.stdout.strip().split(', '))
            print(f"=== System Memory View (nvidia-smi) ===")
            print(f"Used: {used} MB")
            print(f"Free: {free} MB") 
            print(f"Total: {total} MB")
            
            discrepancy = used - reserved
            print(f"=== DISCREPANCY ===")
            print(f"nvidia-smi shows {used} MB used")
            print(f"PyTorch shows {reserved} MB reserved") 
            print(f"Untracked memory: {discrepancy:.2f} MB")
            if discrepancy > 1000:  # More than 1GB untracked
                print("WARNING: Large amount of untracked GPU memory!")
    except Exception as e:
        print(f"Could not get nvidia-smi data: {e}")
    
    # Memory summary (this shows internal fragmentation)
    print(f"=== PyTorch Memory Summary ===")
    try:
        memory_summary = torch.cuda.memory_summary(device)
        for line in memory_summary.split('\n'):
            if 'Active' in line or 'allocated_bytes.all' in line or 'reserved_bytes.all' in line:
                print(line.strip())
    except:
        pass
        
    print("=" * 50)