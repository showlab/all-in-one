from setuptools import setup, find_packages

setup(
    name="AllInOne",
    packages=find_packages(
        exclude=[".dfc", ".vscode", "dataset", "notebooks", "result", "scripts"]
    ),
    version="1.0.0",
    license="MIT",
    description="All in One: Exploring Unified Video-Language Pre-training",
    author="Alex Jinpeng Wang",
    author_email="awinyimgprocess@gmail.com",
    url="https://github.com/fingerrec'",
    keywords=["video and language pretraining"],
    install_requires=["torch", "pytorch_lightning"],
)
