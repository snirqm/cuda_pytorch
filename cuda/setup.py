from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    version="0.1.0",
    name="parallel_mult_cuda",
    ext_modules=[
        CUDAExtension(
            "parallel_mult_cuda",
            [
                "parallel_mult.cpp",
                "parallel_mult_kernel.cu",
            ],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
