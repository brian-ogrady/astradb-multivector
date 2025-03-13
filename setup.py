from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="astra-multivector",
    version="0.1.0",
    author="Brian O'Grady",
    author_email="brian.ogrady@datastax.com",
    description="Multivector CQL Tables using the DataAPI from AstraDB",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/datastax/astra-multivector",
    packages=find_packages(where="libs"),
    package_dir={"": "libs"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
    ],
    python_requires=">=3.12",
    install_requires=[
        "astrapy==2.0.0rc0",
        "langchain>=0.3.19",
        "langchain-nvidia-ai-endpoints>=0.3.9",
        "pydantic>=2.10.6",
        "python-dotenv>=1.0.1",
        "sentence-transformers>=3.4.1",
        "tqdm>=4.67.1",
        "voyageai>=0.3.2",
    ],
    keywords="astradb, vector, database, embeddings, datastax, cassandra",
    project_urls={
        "Bug Tracker": "https://github.com/datastax/astra-multivector/issues",
        "Documentation": "https://github.com/datastax/astra-multivector/blob/main/README.md",
        "Source Code": "https://github.com/datastax/astra-multivector",
    },
)