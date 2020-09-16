import torch
device=torch.device("cuda")

model_points=torch.rand(1000,5,3).to(device)
base=torch.rand(1000,3,3).to(device)
points=torch.rand(1000,1,3).to(device)
pred_t=torch.rand(1000,1,3).to(device)

pred = torch.add(torch.bmm(model_points, base), points + pred_t)


print(pred)

