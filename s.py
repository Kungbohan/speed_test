import time
import torch
from accelerate import Accelerator
from torchvision.models import vit_b_16
import torch.jit as jit
import argparse

# Initialize Accelerator
accelerator = Accelerator()

parser = argparse.ArgumentParser(description='Test ViT performance with different thread counts')
parser.add_argument('--threads', type=int, default=144, help='Number of threads to use')
args = parser.parse_args()

# Model and input settings
model = vit_b_16(weights=None)  # Vision Transformer model without pretrained weights
input_size = (20, 3, 224, 224)  # Input batch size matches number of threads (144 images)
x = torch.randn(input_size)  # Random input tensor
num_threads = args.threads  # Use all available threads on the server

# Set the number of threads for PyTorch
torch.set_num_threads(num_threads)

# Function to process a batch chunk (forward and backward pass)
def process_chunk(model, x_chunk):
    model.train()
    output = model(x_chunk)
    loss = output.mean()  # Dummy loss
    # loss.backward()
    return output

# Function to measure time using multi-threading
def measure_time_multithread(model, x, device):
    model = model.to(device)
    x = x.to(device)
    model.train()
    # Split input into chunks for parallel processing
    batch_size = x.size(0)
    chunk_size = batch_size // num_threads
    x_chunks = [x[i * chunk_size: (i + 1) * chunk_size] for i in range(num_threads)]

    # Warm-up to stabilize measurements
    for _ in range(2):
        futures = [jit.fork(process_chunk, model, chunk) for chunk in x_chunks]
        _ = [jit.wait(f) for f in futures]

    # Forward and backward pass timing
    if device.type == 'cuda':
        torch.cuda.synchronize()  # Sync GPU before timing
    start_time = time.time()
    futures = [jit.fork(process_chunk, model, chunk) for chunk in x_chunks]
    _ = [jit.wait(f) for f in futures]
    if device.type == 'cuda':
        torch.cuda.synchronize()  # Sync GPU after timing
    total_time = time.time() - start_time

    return total_time

# Run test on CPU with multi-threading
cpu_device = torch.device('cpu')
cpu_time = measure_time_multithread(model, x, cpu_device)

# Run test on GPU if available
gpu_device = torch.device('cuda') if torch.cuda.is_available() else None
if gpu_device:
    model = vit_b_16(weights=None)  # Re-initialize model to avoid gradient accumulation
    gpu_time = measure_time_multithread(model, x, gpu_device)
else:
    gpu_time = None
    print("GPU not available. Skipping GPU test.")

# Display results and comparison
print(f"CPU (Multi-threaded, {num_threads} threads) - Total Time: {cpu_time:.4f}s")
if gpu_device:
    print(f"GPU - Total Time: {gpu_time:.4f}s")
    if gpu_time < cpu_time:
        print("GPU is faster by {:.2f}x".format(cpu_time / gpu_time))
    else:
        print("CPU (Multi-threaded) is faster by {:.2f}x".format(gpu_time / cpu_time))