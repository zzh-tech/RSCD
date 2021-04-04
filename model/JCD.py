import torch
import torch.nn as nn
import torch.nn.functional as F
from forward_warp_package import ForwardWarp
from correlation_package import Correlation
from package_core.net_basics import *
from .utils import conv1x1, actFunc, RDNet, ModulatedDeformLayer


def generate_2D_grid(H, W):
    x = torch.arange(0, W, 1).float().cuda()
    y = torch.arange(0, H, 1).float().cuda()

    xx = x.repeat(H, 1)
    yy = y.view(H, 1).repeat(1, W)

    grid = torch.stack([xx, yy], dim=0)

    return grid  # (2,H,W)


class PredImage(nn.Module):
    def __init__(self, in_chs, out_chs):
        super(PredImage, self).__init__()
        self.conv0 = nn.Conv2d(in_chs, out_chs, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = self.conv0(x)
        return out


class ImageEncoder(nn.Module):
    def __init__(self, in_chs, init_chs):
        super(ImageEncoder, self).__init__()
        self.conv0 = conv2d(
            in_planes=in_chs,
            out_planes=init_chs,
            batch_norm=False,
            activation=nn.ReLU(),
            kernel_size=7,
            stride=1
        )
        self.resblocks0 = Cascade_resnet_blocks(in_planes=init_chs, n_blocks=3)  # green block in paper
        self.conv1 = conv2d(
            in_planes=init_chs,
            out_planes=2 * init_chs,
            batch_norm=False,
            activation=nn.ReLU(),
            kernel_size=3,
            stride=2
        )
        self.resblocks1 = Cascade_resnet_blocks(in_planes=2 * init_chs, n_blocks=3)
        self.conv2 = conv2d(
            in_planes=2 * init_chs,
            out_planes=4 * init_chs,
            batch_norm=False,
            activation=nn.ReLU(),
            kernel_size=3,
            stride=2
        )
        self.resblocks2 = Cascade_resnet_blocks(in_planes=4 * init_chs, n_blocks=3)

    def forward(self, x):
        x0 = self.resblocks0(self.conv0(x))
        x1 = self.resblocks1(self.conv1(x0))
        x2 = self.resblocks2(self.conv2(x1))
        return x2, x1, x0


class GSA(nn.Module):
    def __init__(self, n_feats):
        super(GSA, self).__init__()
        activation = 'relu'
        self.F_f = nn.Sequential(
            nn.Linear(3 * n_feats, 6 * n_feats),
            actFunc(activation),
            nn.Linear(6 * n_feats, 3 * n_feats),
            nn.Sigmoid()
        )
        # condense layer
        self.condense = conv1x1(3 * n_feats, n_feats)
        self.act = actFunc(activation)

    def forward(self, cor):
        w = F.adaptive_avg_pool2d(cor, (1, 1)).squeeze()
        if len(w.shape) == 1:
            w = w.unsqueeze(dim=0)
        w = self.F_f(w)
        w = w.reshape(*w.shape, 1, 1)
        out = self.act(self.condense(w * cor))

        return out


class FusionModule(nn.Module):
    def __init__(self, n_feats):
        super(FusionModule, self).__init__()
        self.attention = GSA(n_feats)
        self.deform = ModulatedDeformLayer(n_feats, n_feats, deformable_groups=8)

    def forward(self, x):
        x = self.attention(x)
        return self.deform(x, x)


class ImageDecorder(nn.Module):
    def __init__(self, init_chs, out_chs):
        super(ImageDecorder, self).__init__()
        chs = out_chs
        self.pred_img2 = PredImage(in_chs=4 * init_chs, out_chs=out_chs)
        self.pred_img1 = PredImage(in_chs=2 * init_chs + chs, out_chs=out_chs)
        self.pred_img0 = PredImage(in_chs=init_chs + chs, out_chs=out_chs)
        self.resblocks2 = Cascade_resnet_blocks(in_planes=4 * init_chs, n_blocks=3)
        self.upconv2 = deconv2d(in_planes=4 * init_chs, out_planes=2 * init_chs)
        self.resblocks1 = Cascade_resnet_blocks(in_planes=2 * init_chs + chs, n_blocks=3)
        self.upconv1 = deconv2d(in_planes=2 * init_chs + chs, out_planes=init_chs)
        self.resblocks0 = Cascade_resnet_blocks(in_planes=init_chs + chs, n_blocks=3)
        self.upsample2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.fusion2 = FusionModule(4 * init_chs)
        self.fusion1 = FusionModule(2 * init_chs)
        self.fusion0 = FusionModule(1 * init_chs)

    def forward(self, feat, p_feats, f_feats):
        p_warped2, p_warped1, p_warped0 = p_feats
        f_warped2, f_warped1, f_warped0 = f_feats
        imgs = [None, None, None]
        # level 2
        x2 = torch.cat([p_warped2, feat, f_warped2], dim=1)
        x2 = self.fusion2(x2)
        x2 = self.resblocks2(x2)
        up_x2 = self.upconv2(x2)
        img2 = self.pred_img2(x2)
        imgs[2] = img2
        up_img2 = self.upsample2(img2)
        # level 1
        x1 = torch.cat([p_warped1, up_x2, f_warped1], dim=1)
        x1 = self.fusion1(x1)
        x1 = torch.cat([x1, up_img2], dim=1)
        x1 = self.resblocks1(x1)
        up_x1 = self.upconv1(x1)
        img1 = self.pred_img1(x1)
        imgs[1] = img1
        up_img1 = self.upsample2(img1)
        # level 0
        x0 = torch.cat([p_warped0, up_x1, f_warped0], dim=1)
        x0 = self.fusion0(x0)
        x0 = torch.cat([x0, up_img1], dim=1)
        x0 = self.resblocks0(x0)
        img0 = self.pred_img0(x0)
        imgs[0] = img0

        return imgs


class FlowDecoder(nn.Module):
    def __init__(self, n_inputs, share_encoder, md=[4, 4, 4]):
        super(FlowDecoder, self).__init__()
        self.share_encoder = share_encoder  # True
        self.corr2 = Correlation(pad_size=md[2], kernel_size=1, max_displacement=md[2], stride1=1, stride2=1,
                                 corr_multiply=1)
        self.leakyReLU2 = nn.LeakyReLU(0.1)
        self.corr1 = Correlation(pad_size=md[1], kernel_size=1, max_displacement=md[1], stride1=1, stride2=1,
                                 corr_multiply=1)
        self.leakyReLU1 = nn.LeakyReLU(0.1)
        self.corr0 = Correlation(pad_size=md[0], kernel_size=1, max_displacement=md[0], stride1=1, stride2=1,
                                 corr_multiply=1)
        self.leakyReLU0 = nn.LeakyReLU(0.1)

        # level 2
        dd = np.cumsum([128, 128, 96, 64, 32])
        nd = (2 * md[2] + 1) ** 2
        od = nd * (n_inputs - 1)
        self.conv2_0 = conv2d(od, 128, batch_norm=False, activation=nn.ReLU(), kernel_size=3, stride=1)
        self.conv2_1 = conv2d(od + dd[0], 128, batch_norm=False, activation=nn.ReLU(), kernel_size=3, stride=1)
        self.conv2_2 = conv2d(od + dd[1], 96, batch_norm=False, activation=nn.ReLU(), kernel_size=3, stride=1)
        self.conv2_3 = conv2d(od + dd[2], 64, batch_norm=False, activation=nn.ReLU(), kernel_size=3, stride=1)
        self.conv2_4 = conv2d(od + dd[3], 32, batch_norm=False, activation=nn.ReLU(), kernel_size=3, stride=1)
        self.predict_flow2 = self.predict_flow(od + dd[4])
        self.deconv2 = self.deconv(in_planes=2, out_planes=2, kernel_size=4, stride=2, padding=1)
        self.upfeat2 = self.deconv(in_planes=od + dd[4], out_planes=2, kernel_size=4, stride=2, padding=1)
        # level 1
        dd = np.cumsum([64, 64, 48, 32, 16])
        nd = (2 * md[1] + 1) ** 2
        od = nd * (n_inputs - 1) + 2 + 2
        self.conv1_0 = conv2d(od, 64, batch_norm=False, activation=nn.ReLU(), kernel_size=3, stride=1)
        self.conv1_1 = conv2d(od + dd[0], 64, batch_norm=False, activation=nn.ReLU(), kernel_size=3, stride=1)
        self.conv1_2 = conv2d(od + dd[1], 48, batch_norm=False, activation=nn.ReLU(), kernel_size=3, stride=1)
        self.conv1_3 = conv2d(od + dd[2], 32, batch_norm=False, activation=nn.ReLU(), kernel_size=3, stride=1)
        self.conv1_4 = conv2d(od + dd[3], 16, batch_norm=False, activation=nn.ReLU(), kernel_size=3, stride=1)
        self.predict_flow1 = self.predict_flow(od + dd[4])
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')

    def predict_flow(self, in_planes):
        return nn.Conv2d(in_planes, 2, kernel_size=3, stride=1, padding=1, bias=True)

    def deconv(self, in_planes, out_planes, kernel_size, stride=2, padding=1):
        return nn.ConvTranspose2d(int(in_planes), out_planes, kernel_size, stride, padding, bias=True)

    def forward(self, in0, in1):
        c02, c01, c00 = in0
        c12, c11, c10 = in1
        # level 2
        corr2 = self.leakyReLU2(self.corr2(c12, c02))
        x = torch.cat([self.conv2_0(corr2), corr2], dim=1)
        x = torch.cat([self.conv2_1(x), x], dim=1)
        x = torch.cat([self.conv2_2(x), x], dim=1)
        x = torch.cat([self.conv2_3(x), x], dim=1)
        x = torch.cat([self.conv2_4(x), x], dim=1)
        flow2 = self.predict_flow2(x)
        upflow2 = self.deconv2(flow2)
        upfeat2 = self.upfeat2(x)
        # level 1
        corr1 = self.leakyReLU1(self.corr1(c11, c01))
        x = torch.cat([corr1, upfeat2, upflow2], dim=1)
        x = torch.cat((self.conv1_0(x), x), dim=1)
        x = torch.cat((self.conv1_1(x), x), dim=1)
        x = torch.cat((self.conv1_2(x), x), dim=1)
        x = torch.cat((self.conv1_3(x), x), dim=1)
        x = torch.cat((self.conv1_4(x), x), dim=1)
        flow1 = self.predict_flow1(x) + upflow2 * 2
        flow0 = self.upsample2(flow1)

        return flow0, flow1, flow2


class Model(nn.Module):
    def __init__(self, para=None):
        super(Model, self).__init__()
        self.para = para
        self.in_chs = 3
        self.init_chs = 32
        self.out_chs = 3
        self.n_inputs = 2
        self.share_encoder = True
        self.est_vel = True
        self.pred_middle_gs = True
        self.multi_lvs = True
        self.md = [4, 4, 4]
        self.encoder = ImageEncoder(in_chs=self.in_chs, init_chs=self.init_chs)
        self.middle_net = RDNet(4 * self.init_chs, growth_rate=2 * self.init_chs, num_layer=3, num_blocks=5,
                                activation='relu')
        self.flow_decoder = FlowDecoder(n_inputs=self.n_inputs, share_encoder=self.share_encoder, md=self.md)
        self.img_decoder = ImageDecorder(init_chs=self.init_chs, out_chs=self.out_chs)

    def forward(self, img_rs):

        flow_encoding = img_rs[:, 0, -1:, :, :]
        img_rs = img_rs[:, :, :3, :, :]
        B, N, C, H, W = img_rs.shape
        img_rs = img_rs.reshape(B, N * C, H, W)

        img_rs0 = img_rs[:, 0:self.in_chs, :, :].clone()
        img_rs1 = img_rs[:, self.in_chs:2 * self.in_chs, :, :]
        img_rs2 = img_rs[:, 2 * self.in_chs:3 * self.in_chs, :, :]

        # encoder
        img_feat0 = self.encoder(img_rs0)
        img_feat1 = self.encoder(img_rs1)
        img_feat2 = self.encoder(img_rs2)

        # motion decoder
        p_flow0, p_flow1, p_flow2 = self.flow_decoder(img_feat0, img_feat1)
        f_flow0, f_flow1, f_flow2 = self.flow_decoder(img_feat1, img_feat2)

        # warp image features
        c12, c11, c10 = img_feat1
        mc12 = self.middle_net(c12)

        B, C, H, W = c12.shape
        warper2 = ForwardWarp.create_with_implicit_mesh(B, C, H, W, 2, 0.5)
        if self.est_vel:
            if self.pred_middle_gs:
                t_flow_ref_to_row0 = F.interpolate(flow_encoding, size=(H, W), mode='bilinear', align_corners=False)
            p_flow2 = p_flow2 * t_flow_ref_to_row0
            f_flow2 = f_flow2 * t_flow_ref_to_row0
        p_c12_warped, p_mask2 = warper2(c12, p_flow2)
        f_c12_warped, f_mask2 = warper2(c12, f_flow2)

        B, C, H, W = c11.shape
        warper1 = ForwardWarp.create_with_implicit_mesh(B, C, H, W, 2, 0.5)
        if self.est_vel:
            if self.pred_middle_gs:
                t_flow_ref_to_row0 = F.interpolate(flow_encoding, size=(H, W), mode='bilinear', align_corners=False)
            p_flow1 = p_flow1 * t_flow_ref_to_row0
            f_flow1 = f_flow1 * t_flow_ref_to_row0
        p_c11_warped, p_mask1 = warper1(c11, p_flow1)
        f_c11_warped, f_mask1 = warper1(c11, f_flow1)

        B, C, H, W = c10.shape
        warper0 = ForwardWarp.create_with_implicit_mesh(B, C, H, W, 2, 0.5)
        if self.est_vel:
            if self.pred_middle_gs:
                t_flow_ref_to_row0 = F.interpolate(flow_encoding, size=(H, W), mode='bilinear', align_corners=False)
            p_flow0 = p_flow0 * t_flow_ref_to_row0
            f_flow0 = f_flow0 * t_flow_ref_to_row0
        p_c10_warped, p_mask0 = warper0(c10, p_flow0)
        f_c10_warped, f_mask0 = warper0(c10, f_flow0)

        imgs = self.img_decoder(mc12,
                                [p_c12_warped, p_c11_warped, p_c10_warped],
                                [f_c12_warped, f_c11_warped, f_c10_warped])
        flows = [[p_flow0, p_flow1, p_flow2], [f_flow0, f_flow1, f_flow2]]
        masks = [[p_mask0, p_mask1, p_mask2], [f_mask0, f_mask1, f_mask2]]

        return imgs, masks, flows


if __name__ == '__main__':
    x = torch.randn(4, 2, 3, 256, 320).cuda()
    model = Model().cuda()
    model(x)
