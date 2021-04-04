import torch

def transformation_matrix_multiply(T1, T2):
    if T1.dim() == 2:
        T1=torch.unsqueeze(T1, 0)

    if T2.dim() == 2:
        T2=torch.unsqueeze(T2, 0)

    R1 = T1[:, :, :3]
    t1 = T1[:, :, 3:4]

    R2 = T2[:, :, :3]
    t2 = T2[:, :, 3:4]

    R = torch.bmm(R1, R2)
    t = torch.bmm(R1, t2) + t1

    return torch.cat([R, t], dim=2)

def transformation_matrix_inverse(T):
    if T.dim() == 2:
        T = T.unsqueeze(dim=0)
    R = T[:, :, :3]
    t = T[:, :, 3:4]

    R_inv = R.transpose(2, 1)
    t_inv = torch.bmm(R_inv, t)
    t_inv = -1. * t_inv
    return torch.cat([R_inv, t_inv], dim=2)

def skew_matrix(phi):
    Phi=torch.zeros([phi.shape[0], 3, 3], dtype=phi.dtype, device=phi.device)
    Phi[:, 0, 1] = -phi[:, 2]
    Phi[:, 1, 0] = phi[:, 2]
    Phi[:, 0, 2] = phi[:, 1]
    Phi[:, 2, 0] = -phi[:, 1]
    Phi[:, 1, 2] = -phi[:, 0]
    Phi[:, 2, 1] = phi[:, 0]
    return Phi