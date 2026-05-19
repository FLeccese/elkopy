from setuptools import setup, find_packages

setup(
    name="elkopy",
    version="0.1.0",
    packages=find_packages(include=['elkopy', 'elkopy.*', 'scripts']),

    entry_points={
        'console_scripts': [
            'elkopy = scripts.run_elkopy:main',
            'elkoplot = scripts.run_plot:main'
        ],
    },
)