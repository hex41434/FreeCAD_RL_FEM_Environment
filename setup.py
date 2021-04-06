import setuptools

setuptools.setup(
    name='RLFEM',
    version='0.0.1',
    author='Aida Farahani',
    author_email='aida.farahani@gmail.com ',
    description='Gym-like environment allowing to perform FEM simulations using freecad.',
    platforms='Posix; MacOS X; Windows',
    packages=setuptools.find_packages(where='./src'),
    package_dir={
        '': 'src'
    },
    include_package_data=True,
    install_requires=(
        'numpy',
    ),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Natural Language :: English',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)