import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tensorflow-fewshot", # Replace with your own username
    version="0.0.1",
    author="Aymeric QUESNE",
    author_email="aymeric.quesne@octo.com",
    description="A Python package for few shot learning training and inference in computer vision using Tensorflow.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/aymericq/tensorflow-fewshot",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)