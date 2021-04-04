from setuptools import setup, find_packages
import unittest
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

CUDA_FLAGS = []

ext_modules=[
    CUDAExtension('forward_warp_package_lib', [
        'forward_warp_package_lib/cuda_arithmetic.cu',
        'forward_warp_package_lib/cuda_common.cu',
        'forward_warp_package_lib/cuda_renderer.cu',
        'forward_warp_package_lib/flow_forward_shift.cpp',
        'forward_warp_package_lib/python_bind.cc',
        ]),
    ]

INSTALL_REQUIREMENTS = ['torch']

setup(
    description='forward_warp_package',
    author='Peidong Liu',
    author_email='peidong.liu@inf.ethz.ch',
    license='MIT License',
    version='0.0.1',
    name='forward_warp_package',
    packages=['forward_warp_package', 'forward_warp_package_lib'],
    install_requires=INSTALL_REQUIREMENTS,
    ext_modules=ext_modules,
    cmdclass = {'build_ext': BuildExtension}
)

