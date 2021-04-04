import torch
import math


def generate_2D_grid(H, W):
    x = torch.arange(0, W, 1).float()
    y = torch.arange(0, H, 1).float()

    xx = x.repeat(H, 1)
    yy = y.view(H, 1).repeat(1, W)

    grid = torch.stack([xx, yy], dim=0)

    return grid  # (2,H,W)


def distortion_map(h, w, ref_row, reverse=False):
    grid_row = generate_2D_grid(h, w)[1].float()
    mask = grid_row / (h - 1)
    if reverse:
        mask *= -1.
        ref_row_floor = math.floor(h - 1 - ref_row)
        mask = mask - mask[int(ref_row_floor)] + (h - 1 - ref_row - ref_row_floor) * (1. / (h - 1))
    else:
        ref_row_floor = math.floor(ref_row)
        mask = mask - mask[int(ref_row_floor)] - (ref_row - ref_row_floor) * (1. / (h - 1))

    return mask


def distortion_encoding(x, ref_row):
    assert x.dim() == 4
    n, c, h, w = x.shape
    grid_row = generate_2D_grid(h, w)[1].float()
    mask = grid_row / (h - 1)
    ref_row_floor = math.floor(ref_row)
    mask = mask - mask[int(ref_row_floor)] - (ref_row - ref_row_floor) * (1. / (h - 1))
    mask = mask.reshape(1, 1, h, w)
    x *= mask
    return x, mask


if __name__ == '__main__':
    # x = torch.ones(1, 1, 256, 256).cuda() * 2
    # out, mask = distortion_encoding(x, ref_row=0)
    # print('out:', out)
    # print('mask', mask)
    # x = torch.ones(1, 1, 256, 256).cuda() * 2
    # out, mask = distortion_encoding(x, ref_row=255 / 2)
    # print('out:', out)
    # print('mask', mask)
    # x = torch.ones(1, 1, 256, 256).cuda() * 2
    # out, mask = distortion_encoding(x, ref_row=255)
    # print('out:', out)
    # print('mask', mask)
    ## forward: scanning from up to bottom
    mask = distortion_map(h=256, w=256, ref_row=0)
    print('mask', mask)
    mask = distortion_map(h=256, w=256, ref_row=255 / 2)
    print('mask', mask)
    mask = distortion_map(h=256, w=256, ref_row=255)
    print('mask', mask)
    ## reverse: scanning from bottom to up
    mask = distortion_map(h=256, w=256, ref_row=0, reverse=True)
    print('mask', mask)
    mask = distortion_map(h=256, w=256, ref_row=255 / 2, reverse=True)
    print('mask', mask)
    mask = distortion_map(h=256, w=256, ref_row=255, reverse=True)
    print('mask', mask)
