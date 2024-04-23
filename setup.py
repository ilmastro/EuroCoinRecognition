from setuptools import setup, find_packages

setup(
    name='EuroCoinRecognition',
    version='1.0',
    author='Ilmastro',
    author_email='github.com.snide956@passinbox.com',
    description='A simple project to recognize Euro coin denominations from images.',
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        'numpy',
        'opencv-python-headless',
        'tensorflow',
        'scikit-learn'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    python_requires='>=3.7',
    include_package_data=True,
    zip_safe=False
)
