#!/usr/bin/env python
# -*- coding: utf-8 -*-


from setuptools import setup


with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()



requirements = [
    'Click>=6.0', 
    'pystan', 
 
    #'statsmodels.api'
    # TODO: put package requirements here
]

setup_requirements = [
    'pytest-runner'
]

test_requirements = [
    # TODO: put package test requirements here
    'pytest',
    'patsy', 
    'numpy', 
    'pandas',
]

setup(
    name='loo',
    version='0.1.0',
    description="Calculates the PSIS leave-one-out log predictive densities.",
    long_description=readme + '\n\n' + history,
    author="Devin Etcitty",
    author_email='dce2108@columbia.edu',
    url='https://github.com/detcitty/loo',
    packages=[
        'loo',
    ],
    package_dir={'loo':
                 'loo'},
    entry_points={
        'console_scripts': [
            'loo=loo.cli:main'
        ]
    },
    include_package_data=False,
    install_requires=requirements,
    setup_requires=setup_requirements, 
    license="GNU General Public License v3",
    zip_safe=False,
    keywords='loo',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
    test_suite='tests',
    tests_require=test_requirements
)
