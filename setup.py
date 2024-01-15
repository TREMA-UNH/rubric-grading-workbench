from setuptools import setup, find_packages

setup(
    name="ExamPP",
    version="0.1.0",
    author='Laura Dietz, Ben Gamari, Naghmeh Farzi',
    author_email='dietz@cs.unh.edu',
    description='EXAM++ Answerability Metric plus plus',
    long_description='Uses questions to assess the information content in passages',
    url='https://git.smart-cactus.org/ben/exampp',
    packages=find_packages(),
    install_requires=[
        "pydantic",
        "fuzzywuzzy",
        "nltk",
        "pylatex",         
        "trec-car-tools",  
    ],
    extras_require={
        "dev": [
            "mypy",
            "jedi",
            # Add other development dependencies as needed
        ]
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    entry_points={
        'console_scripts': [
            'exam_grade = exampp.exam_grading:main',
            'exam_post = exampp.exam_post_pipeline:main',
        ],
    },
)
