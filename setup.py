from setuptools import setup, find_packages

setup(
    name="kinfoex",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    description="",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Victorgonl",
    author_email="victorgonl@outlook.com",
    url="https://github.com/Victorgonl/kinfoex/",
    classifiers=[],
    install_requires=[
        "datasets==3.4.1",
        "evaluate==0.4.3",
        "numpy==2.2.4",
        "optuna==4.2.1",
        "packaging==24.2",
        "Pillow==11.1.0",
        "py-cpuinfo==9.0.0",
        "torch==2.6.0",
        "transformers==4.50.0",
        "seqeval==1.2.2",
    ],
)
