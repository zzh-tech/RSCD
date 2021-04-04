#ifndef _FLOW_FORWARD_SHIFT_
#define _FLOW_FORWARD_SHIFT_

#include <torch/torch.h>
#include "cuda_tensor.h"

class Flow_forward_shift
{
public:
    Flow_forward_shift(at::Tensor grid, 
    	                int nChannels_texture,
    	                int kernel_radius,
    	                float kernel_sigma2);
    ~Flow_forward_shift();

    void forward(at::Tensor src_image, 
    	        at::Tensor flow_src_to_tar, 
    	        at::Tensor tar_image_w_I, 
    	        at::Tensor tar_image_w,
                at::Tensor mask);

    void backward(at::Tensor grad_I_target, 
    	        at::Tensor grad_I_source, 
    	        at::Tensor grad_flow);

private:
    CudaTensor<int>* m_grid;

    CudaTensor<float>* m_internal_mem_src_image;
    CudaTensor<float>* m_internal_mem_flow;
    CudaTensor<float>* m_internal_mem_tar_image_w_I;
    CudaTensor<float>* m_internal_mem_tar_image_w;

    CudaTensor<float>* m_internal_mem_grad_I_target;
    CudaTensor<float>* m_internal_mem_grad_I_source;
    CudaTensor<float>* m_internal_mem_grad_flow;    

    CudaTensor<int>* m_pixel_index_map;
    int m_nMaxPixelMaps;

    int m_kernel_radius;
    float m_kernel_sigma2;
};

#endif