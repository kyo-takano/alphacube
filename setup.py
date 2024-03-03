from setuptools import setup

with open('README.md', 'r') as f:
    long_description = f.read()

setup(
    name="alphacube",
    version="0.1.1",
    description="A powerful & flexible Rubik's Cube solver",
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Kyo Takano',
    url='https://alphacube.dev/',
    license="MIT",
    python_requires=">= 3.6",
    packages=["alphacube"],
    install_requires=[
        "torch>=2.0.1",
        "numpy>=1.23.3",
        "rich>=13.0.1",
        "pydantic>=2.0.3",
        "requests>=2.28.2"
    ],
    entry_points={
        "console_scripts": [
            "alphacube = alphacube:cli",
        ],
    },
    classifiers=[  # Add classifiers to provide more information
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        'Programming Language :: Python :: 3 :: Only',
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    project_urls={
        "Documentation": "https://alphacube.dev/docs/index.html",
        "Source": "https://github.com/kyo-takano/alphacube"
    }
)
