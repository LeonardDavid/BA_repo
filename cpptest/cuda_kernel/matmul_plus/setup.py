import setuptools
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# with open("README.md", "r") as fh:
#     long_description = fh.read()

setuptools.setup(
    name='matmulplus',
    version="0.0.1",
    author='LDB',
    author_email='ldb@gmail.com',
    description='matrix mult plus',
    # long_description=long_description,
    # long_description_content_type="text/markdown",
    # url="",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: OS Independent",
    ],
    ext_modules=[
        CUDAExtension('matmul', [
            'main.cpp',
            'kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
