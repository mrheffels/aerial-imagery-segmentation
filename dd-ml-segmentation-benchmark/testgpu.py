import torch

x = torch.cuda.current_device()
print(torch.cuda.device(x))
print(f"Count is {torch.cuda.device_count()}")
print(f"Device name is {torch.cuda.get_device_name(x)}")
print(torch.cuda.is_available())
