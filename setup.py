from setuptools import setup, find_packages

setup(
    name="elkopy",
    version="0.1.0",
    url="https://github.com/FLeccese/elkopy.git",
    licence="MIT",
    author="Francesco Leccese",
    author_email="<francescoleccese4@gmail.com>",
    packages=find_packages(include=['elkopy', 'elkopy.*', 'scripts']),

    entry_points={
        'console_scripts': [
            'elkopy = scripts.run_elkopy:main',
            'elkoplot = scripts.run_plot:main'
        ],
    },
)