# Differential forward warping package

This repository contains a custom pytorch package that forward warp an image with dense displacement field as described in paper [CVPR2020-Deep Shutter Unrolling Network](https://drive.google.com/open?id=14NYguVp129ydRtRzhhU8H8QIiE0coK6x).

## To install:
```
python setup install
```

## Usage:
```
from forward_warp_package import *

B,C,H,W=im.size()
flow = torch.ones([B, 2, H, W]).float().cuda()
warp_operator = ForwardWarp.create_with_implicit_mesh(B, C, H, W, 2, 0.5)
im_warped, mask = warp_operator(im, flow)
```

#### Acknowledgement
If you find our code useful, please acknowledge it appropriately and cite the paper:
```
@InProceedings{Liu2020CVPR,
  author = {Peidong Liu and Zhaopeng Cui and Viktor Larsson and Marc Pollefeys},
  title = {Deep Shutter Unrolling Network},
  booktitle = {Proc. IEEE Conf. on Computer Vision and Pattern	Recognition (CVPR)}},
  year = {2020}
}
```

