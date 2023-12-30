from setuptools import setup, find_packages

setup(
    name='exam_pp',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        # list your dependencies here
    ],
    entry_points={
        'console_scripts': [
            'exam_score = exampp.exam_score:main',
        ],
    },
    author='Laura Dietz, Ben Gamari',
    author_email='dietz@cs.unh.edu',
    description='EXAM++ Answerability Metric plus plus',
    long_description='Uses questions to assess the information content in passages',
    url='https://git.smart-cactus.org/ben/exampp',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)

