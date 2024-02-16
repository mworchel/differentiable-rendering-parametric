from setuptools import setup

# read the contents of your README file (https://packaging.python.org/en/latest/guides/making-a-pypi-friendly-readme/)
from pathlib import Path
this_directory = Path(__file__).parent
readme = (this_directory / "README.md").read_text()

setup(
    name='drpg',
    version='0.0.1',
    description='Differentiable Rendering of Parametric Geometry',
    url='https://github.com/mworchel/differentiable-rendering-parametric',
    author='Markus Worchel',
    author_email='m.worchel@tu-berlin.de',
    license='BSD',
    packages=['drpg'],
    long_description=readme,
    long_description_content_type="text/markdown"
)