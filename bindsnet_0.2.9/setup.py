from setuptools import setup, find_packages

with open("README.md") as f:
    long_description = f.read()

setup(
    name="BindsNET",
    version="0.2.9",
    description="Spiking neural networks for ML in Python",
    license="AGPL-3.0",
    long_description=long_description,
    long_description_content_type="text/markdown",  # This is important!
    url="http://github.com/Hananel-Hazan/bindsnet",
    author="Hananel Hazan, Daniel Saunders, Darpan Sanghavi, Hassaan Khan",
    author_email="hananel@hazan.org.il",
    packages=find_packages(),
    zip_safe=False,
    install_requires=[
        "numpy==1.21.1",
        "torch==1.9.0",
        "torchvision==0.10.0",
        "tensorboardX==2.4",
        "tqdm==4.62.0",
        "matplotlib==3.4.2",
        "gym==0.18.3",
        "scikit-build==0.11.1",
        "scikit_image==0.18.2",
        "scikit_learn==0.24.2",
        "opencv-python==4.5.3.56",
        "pytest==6.2.4",
        "scipy==1.7.1",
        "cython==0.29.24",
        "pandas==1.3.1",
    ],
)
