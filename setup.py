from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='VolumetricSMPL',
    version='1.0.3',  # Bump version to trigger new upload
    packages=['VolumetricSMPL'],
    url='https://github.com/markomih/VolumetricSMPL',
    license='MIT',
    author='Marko Mihajlovic',
    author_email='markomih@inf.ethz.ch',
    description='VolumetricSMPL body model.',
    long_description=long_description,
    long_description_content_type='text/markdown',  # This tells PyPI it's Markdown
    python_requires='>=3.6.0',
    install_requires=[
        'numpy>=1.12.2',
        'trimesh',
        'scikit-image',
        'smplx',
    ],
)