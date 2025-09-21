import torch

print("¿CUDA disponible?:", torch.cuda.is_available())
print("Versión de CUDA:", torch.version.cuda)
print("Número de GPUs detectadas:", torch.cuda.device_count())

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print("  Memoria total (MB):", torch.cuda.get_device_properties(i).total_memory // (1024**2))
