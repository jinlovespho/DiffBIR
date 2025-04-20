import torch 

# # Create a generator with a fixed seed
# g = torch.Generator(device="cuda")  # or "cpu"
# g.manual_seed(42)

# # Generate noise with the generator
# noise = torch.randn((1, 3, 64, 64), generator=g, device="cuda")

# torch.manual_seed(42)

g = torch.Generator(device="cpu")  # or "cpu"
g.manual_seed(42)

for i in range(3):
    
    noise1 = torch.randn((1,1,5,5), generator=g)
    print(noise1)
    print('-'*50)

breakpoint()