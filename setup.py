from setuptools import setup, find_packages


with open("requirements.txt", "r") as req_file:
    requirements = [i.split(' ')[0] for i in req_file.readlines() if not i.startswith("#")]

setup(
    name="pca",
    version="0.1",
    packages=find_packages(),
    # Project uses reStructuredText, so ensure that the docutils get
    # installed or upgraded on the target machine
    install_requires=["docutils>=0.3"] + requirements,
    package_data={},
    # metadata to display on PyPI
    author="Niels Warncke",
    author_email="niels.warncke@gmail.com",
    description="A nice API to use numpy.SVD for PCA dimensionality reduction",
    keywords="PCA",
    classifiers=["License :: OSI Approved :: Python Software Foundation License"]
    # could also include long_description, download_url, etc.
)
