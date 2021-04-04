#include "flow_forward_shift.h"
#include "cuda_common.h"
#include "cuda_arithmetic.h"
#include "cuda_renderer.h"

Flow_forward_shift::Flow_forward_shift(at::Tensor grid, 
                                        int nChannels_image, 
                                        int kernel_radius, 
                                        float kernel_sigma2) 
{
    m_kernel_radius = kernel_radius;
    m_kernel_sigma2 = kernel_sigma2;

    int B = grid.size(0);
    int H = grid.size(2);
    int W = grid.size(3);

    int *temp;
    cudaMalloc(&temp, B * 2 * H * W * sizeof(int));
    from_pytorch_mem_layout(B, 2, H, W, grid.data<int>(), temp);
    m_grid = new CudaTensor<int>(temp, B, 2, H, W);
    m_grid->clone();
    cudaFree(temp);

    m_internal_mem_src_image = new CudaTensor<float>(B, nChannels_image, H, W);
    m_internal_mem_flow = new CudaTensor<float>(B, 2, H, W);
    m_internal_mem_tar_image_w_I = new CudaTensor<float>(B, nChannels_image, H, W);
    m_internal_mem_tar_image_w = new CudaTensor<float>(B, 1, H, W);

    m_internal_mem_grad_I_target = new CudaTensor<float>(B, nChannels_image, H, W);
    m_internal_mem_grad_I_source = new CudaTensor<float>(B, nChannels_image, H, W);
    m_internal_mem_grad_flow = new CudaTensor<float>(B, 2, H, W);

    m_nMaxPixelMaps=(kernel_radius*2) * (kernel_radius*2) * 8;
    m_pixel_index_map = new CudaTensor<int>(B, m_nMaxPixelMaps, H, W);
}

Flow_forward_shift::~Flow_forward_shift() 
{
    delete m_grid;
    delete m_internal_mem_src_image;
    delete m_internal_mem_flow;
    delete m_internal_mem_tar_image_w_I;
    delete m_internal_mem_tar_image_w;

    delete m_internal_mem_grad_I_target;
    delete m_internal_mem_grad_I_source;
    delete m_internal_mem_grad_flow;

    delete m_pixel_index_map;
}

void Flow_forward_shift::forward(at::Tensor src_texture, 
                                at::Tensor flow_src_to_tar, 
                                at::Tensor tar_image_w_I, 
                                at::Tensor tar_image_w,
                                at::Tensor mask) 
{
    from_pytorch_mem_layout(src_texture.size(0),
                            src_texture.size(1),
                            src_texture.size(2),
                            src_texture.size(3),
                            src_texture.data<float>(),
                            m_internal_mem_src_image->data_ptr());
    from_pytorch_mem_layout(flow_src_to_tar.size(0),
                            2,
                            flow_src_to_tar.size(2),
                            flow_src_to_tar.size(3),
                            flow_src_to_tar.data<float>(),
                            m_internal_mem_flow->data_ptr());

    tensor_element_wise_add<int, float, float>(m_grid, m_internal_mem_flow, m_internal_mem_flow);

    m_internal_mem_tar_image_w_I->reset(0);
    m_internal_mem_tar_image_w->reset(0);
    forward_flow_shift_renderer(m_internal_mem_src_image->data_ptr(), 
                                m_internal_mem_flow->data_ptr(),
                                src_texture.size(0),
                                src_texture.size(1),
                                src_texture.size(2),
                                src_texture.size(3),
                                m_kernel_radius,
                                m_kernel_sigma2,
                                m_internal_mem_tar_image_w_I->data_ptr(),
                                m_internal_mem_tar_image_w->data_ptr(),
                                mask.data<float>());

    to_pytorch_mem_layout<float>(tar_image_w_I.size(0),
                                 tar_image_w_I.size(1),
                                 tar_image_w_I.size(2),
                                 tar_image_w_I.size(3),
                                 m_internal_mem_tar_image_w_I->data_ptr(),
                                 tar_image_w_I.data<float>());

    to_pytorch_mem_layout<float>(tar_image_w.size(0),
                                 tar_image_w.size(1),
                                 tar_image_w.size(2),
                                 tar_image_w.size(3),
                                 m_internal_mem_tar_image_w->data_ptr(),
                                 tar_image_w.data<float>());
}

void Flow_forward_shift::backward(at::Tensor grad_I_target, 
                                at::Tensor grad_I_source, 
                                at::Tensor grad_flow)
{
    from_pytorch_mem_layout(grad_I_target.size(0),
                            grad_I_target.size(1),
                            grad_I_target.size(2),
                            grad_I_target.size(3),
                            grad_I_target.data<float>(),
                            m_internal_mem_grad_I_target->data_ptr());

    m_internal_mem_grad_I_source->reset(0);
    m_internal_mem_grad_flow->reset(0);

    backward_flow_shift_renderer(m_internal_mem_src_image->data_ptr(), 
                                m_internal_mem_flow->data_ptr(),
                                grad_I_target.size(0),
                                grad_I_target.size(1),
                                grad_I_target.size(2),
                                grad_I_target.size(3),
                                m_kernel_radius,
                                m_kernel_sigma2,
                                m_internal_mem_tar_image_w_I->data_ptr(),
                                m_internal_mem_tar_image_w->data_ptr(),
                                m_internal_mem_grad_I_target->data_ptr(),
                                m_internal_mem_grad_I_source->data_ptr(),
                                m_internal_mem_grad_flow->data_ptr());

    to_pytorch_mem_layout<float>(grad_I_target.size(0),
                                 grad_I_target.size(1),
                                 grad_I_target.size(2),
                                 grad_I_target.size(3),
                                 m_internal_mem_grad_I_source->data_ptr(),
                                 grad_I_source.data<float>());

    to_pytorch_mem_layout<float>(grad_flow.size(0),
                                 grad_flow.size(1),
                                 grad_flow.size(2),
                                 grad_flow.size(3),
                                 m_internal_mem_grad_flow->data_ptr(),
                                 grad_flow.data<float>());
}

