import os
import sys

from setuptools import setup, find_packages
from setuptools.command.install import install
from codecs import open


VERSION = "0.3.1"


# some excerpts from https://circleci.com/blog/continuously-deploying-python-packages-to-pypi-with-circleci/
def readme():
    with open('README.rst') as f:
        return f.read()

class VerifyVersionCommand(install):
    """Custom command to verify that the git tag matches our version"""
    description = 'verify that the git tag matches our version'

    def run(self):
        tag = os.getenv('CIRCLE_TAG')

        if tag != VERSION:
            info = "Git tag: {0} does not match the version of this app: {1}".format(
                tag, VERSION
            )
            sys.exit(info)

setup(
    name='pipeGraph',
    version=VERSION,
    description="PipeGraph extends the concept of Scikit-Learn's Pipeline tool",
    long_description=readme(),
    url='https://mcasl.github.io/PipeGraph/',
    author='See License.txt file in source package',
    author_email='manuel.castejon@unileon.es',
    license="MIT",
    classifiers=['Intended Audience :: Science/Research',
                 'Intended Audience :: Developers',
                 'License :: OSI Approved :: MIT License',
                 'Topic :: Software Development',
                 'Topic :: Scientific/Engineering',
                 'Operating System :: Microsoft :: Windows',
                 'Operating System :: POSIX',
                 'Operating System :: Unix',
                 'Operating System :: MacOS',
                 'Programming Language :: Python :: 3.6',
    ],

    keywords='scikit-learn pipeline development',
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),

    install_requires=['pytest',
					  'scikit-learn >= 0.17',
					  'networkx',
					  'pandas'
					  ],

    extras_require={
        'dev': ['check-manifest'],
        'test': ['coverage'],
    },
    python_requires='>=3',
    cmdclass={
        'verify': VerifyVersionCommand,
    }

)
