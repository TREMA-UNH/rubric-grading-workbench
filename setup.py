from setuptools import setup

setup(
    name='exampp',
    version='0.1.0',
    packages=['exam_pp'],
    install_requires=['pydantic'],
    package_data = {
        'exam_pp': ['py.typed'],
    },
)