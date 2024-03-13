from setuptools import setup

# read the contents of your README file
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="realtime_operator",
    # other arguments omitted
    long_description=long_description,
    long_description_content_type="text/markdown",
    version='0.1.2',
    description='Reatime Operator for time series data',
    url='https://github.com/ErnstHolger/realtime_operator',
    author='Holger Amort',
    author_email='holgeramort@gmail.com',
    license='MIT',
    packages=['realtime_operator'],
    install_requires=[
          'numpy','numba','matplotlib','pytest','pytest-benchmark'
      ],
      zip_safe=False)
)
