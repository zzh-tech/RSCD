#ifndef _CUDA_RENDERER_H_
#define _CUDA_RENDERER_H_

void forward_flow_shift_renderer(const float* image_src,
                                const float* flow_src_to_tar,
                                const int batch_size,
                                const int nChannelsTexture,
                                const int imageH,
                                const int imageW,
                                const int kernel_radius,
                                const float kernel_sigma2,
                                float* image_tar_w_I,
                                float* image_tar_w,
                                float* mask);

void backward_flow_shift_renderer(const float* image_src,
                                const float* flow_src_to_tar,
                                const int batch_size,
                                const int nChannelsTexture,
                                const int imageH,
                                const int imageW,
                                const int kernel_radius,
                                const float kernel_sigma2,
                                const float* image_tar_w_I,
                                const float* image_tar_w,
                                const float* grad_image_target,
                                float* grad_image_src,
                                float* grad_flow);

#endif