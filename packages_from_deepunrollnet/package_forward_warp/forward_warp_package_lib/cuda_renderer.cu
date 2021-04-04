#include "cuda_renderer.h"
#include <cuda_runtime_api.h>
#include <stdio.h>

template <typename scalar_t>
__global__ void forward_flow_shift_renderer_cuda_kernel(const scalar_t* image_src,
                                                        const scalar_t* flow_src_to_tar,
                                                        const int batch_size,
                                                        const int nChannelsTexture,
                                                        const int imageH,
                                                        const int imageW,
                                                        const int kernel_radius,
                                                        const float kernel_sigma2,
                                                        scalar_t* image_tar_w_I,
                                                        scalar_t* image_tar_w,
                                                        scalar_t* mask)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch_size * imageH * imageW) {
        return;
    }

    const int bn = i / (imageH * imageW);
    
    float flow_x = flow_src_to_tar[2*i];
    float flow_y = flow_src_to_tar[2*i+1];

    int tl_x=flow_x-kernel_radius+0.5;
    int tl_y=flow_y-kernel_radius+0.5;
    int br_x=flow_x+kernel_radius+0.5;
    int br_y=flow_y+kernel_radius+0.5;

    tl_x=(tl_x>=0?tl_x:0);
    tl_y=(tl_y>=0?tl_y:0);
    br_x=(br_x<imageW?br_x:imageW-1);
    br_y=(br_y<imageH?br_y:imageH-1);

    for(int y=tl_y; y<br_y+1; y++)
    {
        for(int x=tl_x; x<br_x+1; x++)
        {
            if(x<0) continue;
            if(x>imageW-1) continue;
            if(y<0) continue;
            if(y>imageH-1) continue;

            float dx = x - flow_x;
            float dy = y - flow_y;
            float d2 = dx*dx+dy*dy;
            float w = expf(-0.5 * d2 / kernel_sigma2);

            int index = bn * (imageH * imageW) + y * imageW + x;
            for (int j=0; j< nChannelsTexture; j++)
            {
                atomicAdd(image_tar_w_I+index*nChannelsTexture + j, w * image_src[i*nChannelsTexture+j]);
            }
            atomicAdd(image_tar_w+index, w);
            mask[index]=1.0;
        }
    }    
}

template <typename scalar_t>
__global__ void backward_flow_shift_renderer_cuda_kernel(const scalar_t* image_src,
                                                        const scalar_t* flow_src_to_tar,
                                                        const int batch_size,
                                                        const int nChannelsTexture,
                                                        const int imageH,
                                                        const int imageW,
                                                        const int kernel_radius,
                                                        const float kernel_sigma2,
                                                        const scalar_t* image_tar_w_I,
                                                        const scalar_t* image_tar_w,
                                                        const scalar_t* grad_image_target,
                                                        scalar_t* grad_image_src,
                                                        scalar_t* grad_flow)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch_size * imageH * imageW) {
        return;
    }

    const int bn = i / (imageH * imageW);
    
    float flow_x = flow_src_to_tar[2*i];
    float flow_y = flow_src_to_tar[2*i+1];

    int tl_x=flow_x-kernel_radius+0.5;
    int tl_y=flow_y-kernel_radius+0.5;
    int br_x=flow_x+kernel_radius+0.5;
    int br_y=flow_y+kernel_radius+0.5;

    tl_x=(tl_x>=0?tl_x:0);
    tl_y=(tl_y>=0?tl_y:0);
    br_x=(br_x<imageW?br_x:imageW-1);
    br_y=(br_y<imageH?br_y:imageH-1);

    for(int y=tl_y; y<br_y+1; y++)
    {
        for(int x=tl_x; x<br_x+1; x++)
        {
            if(x<0) continue;
            if(x>imageW-1) continue;
            if(y<0) continue;
            if(y>imageH-1) continue;

            float dx = x - flow_x;
            float dy = y - flow_y;
            float d2 = dx*dx+dy*dy;
            float w = expf(-0.5 * d2 / kernel_sigma2);

            int index = bn * (imageH * imageW) + y * imageW + x;
            float It_w = image_tar_w[index];

            float Dw_DflowX = w*(-0.5)/kernel_sigma2*(-2.0)*dx;
            float Dw_DflowY = w*(-0.5)/kernel_sigma2*(-2.0)*dy;

            for(int j=0; j < nChannelsTexture; j++)
            {
                float De_DItj = grad_image_target[index*nChannelsTexture+j];
                float Is_j = image_src[i*nChannelsTexture + j];
                float It_w_Ij = image_tar_w_I[index*nChannelsTexture+j];
                float DItj_Dw = (Is_j*It_w - It_w_Ij)/(It_w*It_w+1e-8);
                float De_DflowX = De_DItj * DItj_Dw * Dw_DflowX;
                float De_DflowY = De_DItj * DItj_Dw * Dw_DflowY;
                // accumulate 
                grad_flow[i*2]+=De_DflowX;
                grad_flow[i*2+1]+=De_DflowY;  

                grad_image_src[i*nChannelsTexture+j] += De_DItj * w / (It_w+1e-8);
            }
        }
    }
}

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
                                float* mask)
{
    const int threads = 512;
    const dim3 blocks ((batch_size * imageH * imageW - 1) / threads + 1);

    forward_flow_shift_renderer_cuda_kernel<float><<<blocks, threads>>>(image_src,
        flow_src_to_tar,
        batch_size,
        nChannelsTexture,
        imageH,
        imageW,
        kernel_radius,
        kernel_sigma2,
        image_tar_w_I,
        image_tar_w,
        mask);

    cudaDeviceSynchronize();
}

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
                                float* grad_flow)
{
    const int threads = 512;
    const dim3 blocks ((batch_size * imageH * imageW - 1) / threads + 1);

    backward_flow_shift_renderer_cuda_kernel<float><<<blocks, threads>>>(image_src,
        flow_src_to_tar,
        batch_size,
        nChannelsTexture,
        imageH,
        imageW,
        kernel_radius,
        kernel_sigma2,
        image_tar_w_I,
        image_tar_w,
        grad_image_target,
        grad_image_src,
        grad_flow);

    cudaDeviceSynchronize();
}

