import torch
from model import light_net 

def main():
    net = light_net()
    net.eval()          

    x = torch.rand(1, 3, 256, 256)    # random 256x256 RGB image

    with torch.no_grad():
        y, xr, xr1 = net(x)

    print("enhanced:", y.shape, y.min().item(), y.max().item())
    print("alpha_stack:", xr.shape)
    print("beta_stack :", xr1.shape)

    assert y.shape == x.shape
    assert xr.shape[1] == 21 and xr1.shape[1] == 21 

if __name__ == "__main__":
    main()
