from setuptools import setup
setup(
    name = 'package',
    version = '0.1.5',
    description = 'Useful functions for training and deployment',
    author_email = 'oamenmodupe@gmail.com',
    packages = ['package.data_retrieval', 'package.eda'],
    install_requires = ['numpy', 'pandas', 'scikit-learn', 'matplotlib', 'mlflow', 'boto3','seaborn' ]
)

