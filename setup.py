from setuptools import setup

setup(
    name="pytorch_tiramisu",
    version="1.0",
    description="Pytorch Tiramisu Neural Network",
    author="Nicolas Pielawski",
    author_email="nicolas@pielawski.fr",
    packages=["models", "models.layers"],
    install_requires=[
        "torch",
    ],
)
