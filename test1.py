import torch 


count=0
gen = torch.Generator('cuda')


while count < 10:
    count +=1 
    print('COUNT: ', count)

    gen.manual_seed(25)

    for i in range(3):
        pure_noise = torch.randn((1, 4, 2, 2), generator=gen, device='cuda', dtype=torch.float32)
        print('1: ', pure_noise)

    print('----------------------')
    gen.manual_seed(25)
    
    for i in range(3):
        pure_noise = torch.randn((1, 4, 2, 2), device='cuda', generator=gen, dtype=torch.float32)
        print('2:',pure_noise)
    
    print('========================')