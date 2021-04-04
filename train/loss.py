from torch.nn.modules.loss import _Loss
from package_core.losses import VariationLoss, L1Loss, PerceptualLoss
from package_core.image_proc import *


# L2 loss
def MSE(para):
    return nn.MSELoss()


# L1 loss
def L1(para):
    return nn.L1Loss()


def MaskedL1(para):
    return L1Loss()


# Variance loss
def Variation(para):
    return VariationLoss(nc=2)


# gradient loss
class L1GradientLoss(_Loss):
    def __init__(self, para):
        super(L1GradientLoss, self).__init__()
        self.get_grad = Gradient()
        self.L1 = nn.L1Loss()

    def forward(self, x, y):
        grad_x = self.get_grad(x)
        grad_y = self.get_grad(y)
        loss = self.L1(grad_x, grad_y)
        return loss


class Gradient(nn.Module):
    def __init__(self):
        super(Gradient, self).__init__()
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False).cuda()
        self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False).cuda()

    def forward(self, x):
        x0 = x[:, 0]
        x1 = x[:, 1]
        x2 = x[:, 2]
        x0_v = F.conv2d(x0.unsqueeze(1), self.weight_v, padding=2)
        x0_h = F.conv2d(x0.unsqueeze(1), self.weight_h, padding=2)

        x1_v = F.conv2d(x1.unsqueeze(1), self.weight_v, padding=2)
        x1_h = F.conv2d(x1.unsqueeze(1), self.weight_h, padding=2)

        x2_v = F.conv2d(x2.unsqueeze(1), self.weight_v, padding=2)
        x2_h = F.conv2d(x2.unsqueeze(1), self.weight_h, padding=2)

        x0 = torch.sqrt(torch.pow(x0_v, 2) + torch.pow(x0_h, 2) + 1e-6)
        x1 = torch.sqrt(torch.pow(x1_v, 2) + torch.pow(x1_h, 2) + 1e-6)
        x2 = torch.sqrt(torch.pow(x2_v, 2) + torch.pow(x2_h, 2) + 1e-6)

        x = torch.cat([x0, x1, x2], dim=1)
        return x


# Charbonnier loss
class L1_Charbonnier_loss(_Loss):
    """L1 Charbonnierloss."""

    def __init__(self, para):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-3

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps * self.eps)
        loss = torch.mean(error)
        return loss


class L1_Charbonnier_loss_color(_Loss):
    """L1 Charbonnierloss."""

    def __init__(self, para):
        super(L1_Charbonnier_loss_color, self).__init__()
        self.eps = 1e-3

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        diff_sq = diff * diff
        # print(diff_sq.shape)
        diff_sq_color = torch.mean(diff_sq, 1, True)
        # print(diff_sq_color.shape)
        error = torch.sqrt(diff_sq_color + self.eps * self.eps)
        loss = torch.mean(error)
        return loss


def Perceptual(para):
    return PerceptualLoss(loss=nn.L1Loss())


# parse loss parameters
def loss_parse(loss_str):
    ratios = []
    losses = []
    str_temp = loss_str.split('|')
    for item in str_temp:
        substr_temp = item.split('*')
        ratios.append(float(substr_temp[0]))
        losses.append(substr_temp[1])
    return ratios, losses


# Training loss
class Loss(nn.Module):
    def __init__(self, para):
        super(Loss, self).__init__()
        ratios, losses = loss_parse(para.loss)
        self.losses_name = losses
        self.ratios = ratios
        self.losses = []
        self.downsample2 = nn.AvgPool2d(2, stride=2)
        for loss in losses:
            # module = import_module('train.loss')
            # self.losses.append(getattr(module, loss)(para).cuda())
            loss_fn = eval('{}(para)'.format(loss))
            self.losses.append(loss_fn)

    def forward(self, x, y, flow=None, valid_flag=False):
        if len(x.shape) == 5:
            b, n, c, h, w = x.shape
            x = x.reshape(b * n, c, h, w)
            y = y.reshape(b * n, c, h, w)
        losses = {}
        loss_all = None
        for i in range(len(self.losses)):
            if valid_flag == True and self.losses_name[i] == 'GAN':
                loss_sub = self.ratios[i] * self.losses[i](x, y, valid_flag)
            elif self.losses_name[i] == 'Variation':
                loss_sub = self.ratios[i] * self.losses[i](flow)
            else:
                loss_sub = self.ratios[i] * self.losses[i](x, y)
            losses[self.losses_name[i]] = loss_sub
            if loss_all == None:
                loss_all = loss_sub
            else:
                loss_all += loss_sub
        losses['all'] = loss_all

        return losses

    def rscd_forward(self, imgs, labels, masks, flows):
        losses = {}

        # reshape tensors
        if len(labels.shape) == 5:
            b, n, c, h, w = labels.shape
            labels = labels.reshape(b * n, c, h, w)
        gts = [labels, ]

        # create multilevel groundtruth
        for i in range(1, len(imgs)):
            labels = self.downsample2(labels.clone())
            gts.append(labels)

        # calculate each loss
        loss_all = None
        for i in range(len(self.losses)):
            sub_loss = None
            for level in range(len(imgs)):
                if self.losses_name[i] == 'Variation':
                    loss_temp = self.ratios[i] * self.losses[i](flows[0][level], mean=True)
                    if len(flows) == 2:
                        loss_temp += self.ratios[i] * self.losses[i](flows[1][level], mean=True)
                elif self.losses_name[i] == 'Perceptual':
                    loss_temp = self.ratios[i] * self.losses[i].get_loss(imgs[level], gts[level])
                else:
                    loss_temp = self.ratios[i] * self.losses[i](imgs[level], gts[level])
                if sub_loss == None:
                    sub_loss = loss_temp
                else:
                    sub_loss += loss_temp
            losses[self.losses_name[i]] = sub_loss
            if loss_all == None:
                loss_all = sub_loss
            else:
                loss_all += sub_loss
        losses['all'] = loss_all

        return losses
