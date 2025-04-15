import torch
from torchvision.ops import roi_align, roi_pool


t1 = torch.rand(1,3,4,4)

box = torch.tensor([[0, 0, 3, 3]]).float()

t2 = roi_align(t1, [box], output_size=(2,2), spatial_scale=1.0)



# Example input tensor (batch_size=1, channels=1, height=4, width=4)
t1 = torch.tensor([[[[1, 2, 3, 4],  
                     [5, 6, 7, 8],  
                     [9, 10, 11, 12],  
                     [13, 14, 15, 16]]]], dtype=torch.float)

# Define the region of interest (ROI) as [batch_index, x1, y1, x2, y2]
# Here, we select the region [0, 0, 0, 3, 3] for the first (and only) image
box = torch.tensor([[0, 0, 0, 3, 3]], dtype=torch.float)

# Apply roi_pool with a 2x2 output size and spatial_scale=1.0
output = roi_pool(t1, box, output_size=(2, 2), spatial_scale=1.0)


breakpoint()