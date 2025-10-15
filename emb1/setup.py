from setuptools import setup, find_packages
setup(
    name="emb1",
    packages=find_packages(),
    install_requires=["pyannote.database >= 4.0"],
    entry_points={
        "pyannote.database.loader": [
            # load embedding files 
            # with your_package.loader.Emb1Loader
            ".emb1 = emb1.loader:Emb1Loader",
        ],
    }
)
