import torch
import torch.nn as nn
import time

# --- Configuration ---
# You can adjust these values to see how performance changes.
# Larger values will make the difference between CPU and GPU more apparent.
MATRIX_SIZE = 4096  # Dimension for the square matrices
NUM_ITERATIONS = 100   # Number of times to repeat the multiplication

def get_device():
    """
    Determines the best available device for PyTorch computations.
    It checks for NVIDIA's CUDA, Apple's Metal Performance Shaders (MPS),
    and defaults to CPU if neither is available.
    """
    if torch.cuda.is_available():
        print("CUDA (NVIDIA GPU) is available. Using GPU.")
        return torch.device("cuda")
    # Check for Apple Silicon (M1/M2/M3 chips)
    elif torch.backends.mps.is_available():
        print("MPS (Apple Silicon GPU) is available. Using GPU.")
        return torch.device("mps")
    else:
        print("No GPU detected. Running on CPU.")
        return torch.device("cpu")

def benchmark(device_name, size, iterations):
    """
    Performs a benchmark of matrix multiplication on a specified device.

    Args:
        device_name (torch.device): The device (CPU or GPU) to run the benchmark on.
        size (int): The dimension of the square matrices to be multiplied.
        iterations (int): The number of times to perform the multiplication.

    Returns:
        float: The total time taken for the benchmark in seconds.
    """
    print(f"\n--- Starting benchmark on: {str(device_name).upper()} ---")

    # Create two large random matrices on the specified device.
    # Using torch.randn allocates the memory directly on the target device.
    try:
        a = torch.randn(size, size, device=device_name)
        b = torch.randn(size, size, device=device_name)
    except Exception as e:
        print(f"Error creating tensors on {device_name}: {e}")
        print("This can happen if the GPU runs out of memory.")
        print("Try reducing the MATRIX_SIZE.")
        return float('inf')


    # GPU Warm-up: The first operation on a GPU can have some overhead
    # for kernel loading, etc. Performing a single dummy operation first
    # ensures a more accurate timing of the actual benchmark loop.
    if device_name.type != 'cpu':
        print("Warming up the GPU...")
        torch.matmul(a, b)
        # For CUDA, synchronize to make sure the warm-up operation is complete
        # before starting the timer.
        if device_name.type == 'cuda':
            torch.cuda.synchronize()

    # Start the timer
    start_time = time.time()

    # The core of the benchmark: a loop of matrix multiplications
    for _ in range(iterations):
        c = torch.matmul(a, b)

    # For GPU operations, the code execution is asynchronous.
    # The CPU might reach this point before all GPU tasks are finished.
    # torch.cuda.synchronize() or torch.mps.synchronize() is a blocking call
    # that waits for all previously queued tasks on the GPU to complete.
    # This is crucial for getting an accurate end time.
    if device_name.type == 'cuda':
        torch.cuda.synchronize()
    elif device_name.type == 'mps':
        torch.mps.synchronize() # For Apple Silicon

    # Stop the timer
    end_time = time.time()

    total_time = end_time - start_time
    print(f"Benchmark on {str(device_name).upper()} completed.")
    return total_time

if __name__ == "__main__":
    # 1. Determine the best available GPU-like device
    gpu_device = get_device()

    # 2. Benchmark on CPU
    cpu_device = torch.device("cpu")
    cpu_time = benchmark(cpu_device, MATRIX_SIZE, NUM_ITERATIONS)
    print(f"Total CPU Time: {cpu_time:.4f} seconds")

    # 3. Benchmark on GPU (if available)
    if gpu_device.type != 'cpu':
        gpu_time = benchmark(gpu_device, MATRIX_SIZE, NUM_ITERATIONS)
        print(f"Total GPU Time: {gpu_time:.4f} seconds")

        # 4. Calculate and display the performance difference
        if gpu_time > 0 and cpu_time > 0:
            speedup = cpu_time / gpu_time
            print(f"\n--- Results ---")
            print(f"GPU was approximately {speedup:.2f} times faster than the CPU.")
    else:
        print("\n--- Results ---")
        print("Only CPU was available. Cannot compare performance.")
