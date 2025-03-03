import os
import argparse
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import time
from torch.utils.data import Dataset, DataLoader, DistributedSampler

import sys
print("Using Python executable:", sys.executable)

from torchvision.models import vit_b_16

class RandomImageDataset(Dataset):
    def __init__(self, num_samples, image_size=(3, 224, 224)):
        self.num_samples = num_samples
        self.image_size = image_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate a random image tensor.
        image = torch.randn(self.image_size)
        # Dummy label for a 1000-class classification task.
        label = torch.randint(0, 1000, (1,)).item()
        return image, label

def setup():
    # Initializes the default process group using environment variables.
    dist.init_process_group(backend='gloo', init_method='env://')

def cleanup():
    dist.destroy_process_group()

def main():
    setup()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    parser = argparse.ArgumentParser(
        description="Benchmark NN forward/backward speed on CPU using Accelerate"
    )
    parser.add_argument("--threads", type=int, default=4, help="Number of CPU threads to use")
    parser.add_argument("--iter", type=int, default=2, help="Number of iterations to run")
    parser.add_argument("--gpu", action = 'store_true', help="Use GPU")
    args = parser.parse_args()
    
    print(f"Using Threads: {args.threads}, Rank={rank}, World_size={world_size}")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(args.threads)
    device = torch.device("cuda" if args.gpu else "cpu")
    print(f"=== Using device: {device} ===")
    
    # Report OS CPU count for reference.
    cpu_count = os.cpu_count()
    # Get number of threads PyTorch is using for intra-operator parallelism.
    intra_threads = torch.get_num_threads()
    # Get number of threads fr inter-operator parallelism.
    inter_threads = torch.get_num_interop_threads()
    print(f"Rank {rank}/{world_size}: OS reports {cpu_count} CPU cores.")
    print(f"Rank {rank}/{world_size}: torch.get_num_threads() (intra threads): {intra_threads}")
    print(f"Rank {rank}/{world_size}: torch.get_num_interop_threads() (inter threads): {inter_threads}")

    # Create the ViT model and wrap it in DDP.
    model = vit_b_16(weights=None)  # Using untrained ViT for demo.
    model = model.to(device)
    ddp_model = DDP(model)

    # Create a random dataset and use DistributedSampler.
    num_samples = 128
    batch_size = 32
    dataset = RandomImageDataset(num_samples)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()


    if dist.is_initialized():
        dist.barrier()
    # Measure forward and backward times.
    num_iters = 2
    total_forward_time = 0.0
    total_backward_time = 0.0

    for _ in range(num_iters):
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            
            # Measure forward pass time.
            start_forward = time.time()
            outputs = ddp_model(images)
            end_forward = time.time()
            
            loss = criterion(outputs, torch.tensor(labels, dtype=torch.long))
            
            # Measure backward pass time.
            start_backward = time.time()
            loss.backward()
            end_backward = time.time()
            
            optimizer.step()
            
            total_forward_time += (end_forward - start_forward)
            total_backward_time += (end_backward - start_backward)

    avg_forward = total_forward_time / num_iters
    avg_backward = total_backward_time / num_iters

    print(f"Rank {rank}-{world_size} - Avg forward time per iteration: {avg_forward:.4f} sec")
    print(f"Rank {rank}-{world_size} - Avg backward time per iteration: {avg_backward:.4f} sec")

    cleanup()   

if __name__ == "__main__":
    main()

