from setuptools import setup, find_packages

setup(
    name="kinfoex",
    version="1.0.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    description="",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Victorgonl",
    author_email="victorgonl@outlook.com",
    url="https://github.com/Victorgonl/kinfoex/",
    classifiers=[],
)
