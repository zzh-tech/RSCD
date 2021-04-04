import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def white_balance(img):
    img = (img*255.).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(img[:, :, 1])
    avg_b = np.average(img[:, :, 2])
    img[:, :, 1] = img[:, :, 1] - ((avg_a - 128) * (img[:, :, 0] / 255.0) * 1.1)
    img[:, :, 2] = img[:, :, 2] - ((avg_b - 128) * (img[:, :, 0] / 255.0) * 1.1)
    img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
    img = img.astype(np.float)/255.
    return img

def warp_image_flow(ref_image, flow):
    [B, _, H, W] = ref_image.size()
    
    # mesh grid 
    xx = torch.arange(0, W).view(1,-1).repeat(H,1)
    yy = torch.arange(0, H).view(-1,1).repeat(1,W)
    xx = xx.view(1,1,H,W).repeat(B,1,1,1)
    yy = yy.view(1,1,H,W).repeat(B,1,1,1)
    grid = torch.cat((xx,yy),1).float()

    if ref_image.is_cuda:
        grid = grid.cuda()

    flow_f = flow + grid
    flow_fx = flow_f[:, 0, :, :] 
    flow_fy = flow_f[:, 1, :, :]

    with torch.no_grad():
        mask_x = ~((flow_fx < 0) | (flow_fx > (W - 1)))
        mask_y = ~((flow_fy < 0) | (flow_fy > (H - 1)))
        mask = mask_x & mask_y
        mask = mask.unsqueeze(1)

    flow_fx = flow_fx / float(W) * 2. - 1.
    flow_fy = flow_fy / float(H) * 2. - 1.

    flow_fxy = torch.stack([flow_fx, flow_fy], dim=-1)
    img = torch.nn.functional.grid_sample(ref_image, flow_fxy, padding_mode='zeros') 
    return img, mask

def warp_image(depth_cur, T_cur2ref, K, img_ref, crop_tl_h=0, crop_tl_w=0):
    B,_, H, W = depth_cur.size()
    fx, fy, cx, cy=K[:,0,0], K[:,1,1], K[:,0,2], K[:,1,2]
    fx = fx.unsqueeze(-1)
    fy = fy.unsqueeze(-1)
    cx = cx.unsqueeze(-1)
    cy = cy.unsqueeze(-1)

    x_ref = torch.arange(0, W, 1).float().cuda() + crop_tl_w
    y_ref = torch.arange(0, H, 1).float().cuda() + crop_tl_h

    x_ref = x_ref.unsqueeze(0).repeat(B,1)
    y_ref = y_ref.unsqueeze(0).repeat(B,1)

    x_ref = (x_ref - cx)/fx
    y_ref = (y_ref - cy)/fy

    x_ref = x_ref.unsqueeze(1)
    y_ref = y_ref.unsqueeze(-1)

    xx_ref = x_ref.repeat(1, H, 1).unsqueeze(1)
    yy_ref = y_ref.repeat(1, 1, W).unsqueeze(1)
    ones = torch.ones_like(xx_ref)

    p3d_ref = torch.cat([xx_ref, yy_ref, ones], dim=1)*depth_cur
    ones = torch.ones_like(depth_cur)

    p4d_ref = torch.cat([p3d_ref, ones], dim=1)
    p4d_ref = p4d_ref.view(B,4,-1)

    p3d_cur = T_cur2ref.bmm(p4d_ref)
    p3d_cur = p3d_cur/(p3d_cur[:,2,:].unsqueeze(dim=1) + 1e-8)
    p2d_cur = K.clone().bmm(p3d_cur)[:,:2,:]

    p2d_cur[:,0,:]=p2d_cur[:,0,:] - crop_tl_w
    p2d_cur[:,1,:]=p2d_cur[:,1,:] - crop_tl_h

    # normalize
    p2d_cur[:,0,:] = 2.0 * (p2d_cur[:,0,:] - W * 0.5 + 0.5) / (W - 1.)
    p2d_cur[:,1,:] = 2.0 * (p2d_cur[:,1,:] - H * 0.5 + 0.5) / (H - 1.)

    p2d_cur = p2d_cur.permute(0, 2, 1)
    p2d_cur = p2d_cur.view(B,H,W,2)

    with torch.no_grad():
        mask_x = ~((p2d_cur[:,:,:,0] < -1.) | (p2d_cur[:,:,:,0] > 1.))
        mask_y = ~((p2d_cur[:,:,:,1] < -1.) | (p2d_cur[:,:,:,1] > 1.))
        mask = mask_x & mask_y
        mask = mask.unsqueeze(1)

    syn_ref_image = torch.nn.functional.grid_sample(img_ref, p2d_cur, padding_mode='zeros')    
    return syn_ref_image, mask.float()

class Grid_gradient_central_diff():
    def __init__(self, nc, padding=True, diagonal=False):
        self.conv_x = nn.Conv2d(nc, nc, kernel_size=2, stride=1, bias=False)
        self.conv_y = nn.Conv2d(nc, nc, kernel_size=2, stride=1, bias=False)
        self.conv_xy = None
        if diagonal:
            self.conv_xy = nn.Conv2d(nc, nc, kernel_size=2, stride=1, bias=False)
    
        self.padding=None
        if padding:
            self.padding = nn.ReplicationPad2d([0,1,0,1])

        fx = torch.zeros(nc, nc, 2, 2).float().cuda()
        fy = torch.zeros(nc, nc, 2, 2).float().cuda()
        if diagonal:
            fxy = torch.zeros(nc, nc, 2, 2).float().cuda()
        
        fx_ = torch.tensor([[1,-1],[0,0]]).cuda()
        fy_ = torch.tensor([[1,0],[-1,0]]).cuda()
        if diagonal:
            fxy_ = torch.tensor([[1,0],[0,-1]]).cuda()

        for i in range(nc):
            fx[i, i, :, :] = fx_
            fy[i, i, :, :] = fy_
            if diagonal:
                fxy[i,i,:,:] = fxy_
            
        self.conv_x.weight = nn.Parameter(fx)
        self.conv_y.weight = nn.Parameter(fy)
        if diagonal:
            self.conv_xy.weight = nn.Parameter(fxy)

    def __call__(self, grid_2d):
        _image = grid_2d
        if self.padding is not None:
            _image = self.padding(_image)
        dx = self.conv_x(_image)
        dy = self.conv_y(_image)

        if self.conv_xy is not None:
            dxy = self.conv_xy(_image)
            return dx, dy, dxy
        return dx, dy

class RandomScaleCrop(object):
    """Randomly zooms images up to 15% and crop them to keep same size as before."""
    def __call__(self, images, intrinsics):
        assert intrinsics is not None
        output_intrinsics = intrinsics.clone()

        _, _, in_h, in_w = images.size()
        x_scaling, y_scaling = np.random.uniform(1, 1.15, 2)
        scaled_h, scaled_w = int(in_h * y_scaling), int(in_w * x_scaling)

        output_intrinsics[:, 0, 0] *= x_scaling
        output_intrinsics[:, 0, 2] *= x_scaling
        output_intrinsics[:, 1, 1] *= y_scaling
        output_intrinsics[:, 1, 2] *= y_scaling
        scaled_images = F.interpolate(images, size=(scaled_h, scaled_w), mode='bilinear', align_corners=False)
       
        offset_y = np.random.randint(scaled_h - in_h + 1)
        offset_x = np.random.randint(scaled_w - in_w + 1)
        cropped_images = scaled_images[:, :, offset_y:offset_y + in_h, offset_x:offset_x + in_w]

        output_intrinsics[:, 0, 2] -= offset_x
        output_intrinsics[:, 1, 2] -= offset_y

        return cropped_images, output_intrinsics

if __name__=='__main__':
    from skimage import io
    import cv2
    image = io.imread('/home/peidong/leonhard/project/infk/cvg/liup/mydata/KITTI/odometry/resized/832x256/test/09/image_2/001545.jpg')
    image = torch.from_numpy(image.transpose(2,0,1)).float()/255.
    image = image.unsqueeze(0)

    intrinsics = torch.eye(3).float().unsqueeze(0)
    cropped_images, intrinsics = RandomScaleCrop()(image, intrinsics)

    cv2.imshow('orig', image.numpy().transpose(0,2,3,1)[0])
    cv2.imshow('crop', cropped_images.numpy().transpose(0,2,3,1)[0])
    cv2.waitKey(0)
