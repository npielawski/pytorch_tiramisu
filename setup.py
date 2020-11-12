import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tiramisu",
    version="1.0",
    description="Pytorch Tiramisu Neural Network",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Nicolas Pielawski",
    author_email="nicolas@pielawski.fr",
    packages=setuptools.find_packages(),
    install_requires=["torch"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
