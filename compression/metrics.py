from typing import Dict, Tuple
import torch


def size_of_model(model: torch.nn.Module) -> float:
    """Returns size of model in Mb."""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    return (param_size + buffer_size) / 1e6


def number_of_model_params(model: torch.nn.Module) -> int:
    """Returns number of model parameters."""
    return sum(p.numel() for p in model.parameters())


def profile(model: torch.nn.Module, input_size, device='cuda', with_flops=True):
    model.to(device)
    model.eval()
    image = torch.randn(input_size, device=device)
    model(image)  # warm-up
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        with_flops=with_flops
    ) as prof:
        with torch.profiler.record_function("model_inference"):
            model(image)
    return prof


def flop(profiler) -> Dict:
    flops = {}
    for e in profiler.events():
        if e.flops != 0:
            if e.key not in flops.keys():
                flops[e.key] = {'flops': e.flops, 'n': 1}
            else:
                flops[e.key]['flops'] += e.flops
                flops[e.key]['n'] += 1
    return flops


def inference_metrics(
        model: torch.nn.Module,
        input_size,
        device='cuda',
        n=10
) -> Tuple[float, float, float, float, float, float]:
    m_params = number_of_model_params(model) / 10**6
    size = size_of_model(model)
    profiler = profile(model, input_size, device)
    gflop = sum([e.flops for e in profiler.events()]) / 10**9
    cpu_time, gpu_time = mean_inference_time(model, input_size, n, device)
    return m_params, size, gflop, cpu_time + gpu_time, cpu_time, gpu_time


def mean_inference_time(
        model: torch.nn.Module,
        input_size,
        n: int = 10,
        device='cuda'
) -> Tuple[float, float]:
    cpu_time = []
    gpu_time = []
    for i in range(n):
        profiler = profile(model, input_size, device, with_flops=False)
        inference = profiler.events()[0]
        assert inference.name == 'model_inference'
        cpu_time.append(inference.cpu_time / 10**3)
        gpu_time.append(inference.cuda_time / 10**3)
    return sum(cpu_time) / n, sum(gpu_time) / n
