from setuptools import setup, find_packages

setup(
    name='src',
    version='0.0.1',
    install_requires=[
        'requests',
        'importlib-metadata; python_version<"3.10"',
        'ultralytics',
        'torch>=1.7.0',
        'torchvision>=0.8.1',
    ],
    packages=find_packages(),
)
