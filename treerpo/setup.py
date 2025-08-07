
from pathlib import Path
from setuptools import find_packages, setup

ROOT = Path(__file__).parent


long_description = (ROOT / "README.md").read_text(encoding="utf-8")

def _read_requirements():
    req_path = ROOT / "requirements.txt"
    if not req_path.exists():
        return []
    lines = (line.strip() for line in req_path.read_text().splitlines())
    return [ln for ln in lines if ln and not ln.startswith("#")]

setup(
    name="treerpo",
    version="0.1.0",
    author="Omri Sapir",
    description="TreeRPO: Hierarchical Credit Assignment for Reasoning in Language Models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/<your-username>/treerpo",  # update before publishing
    license="Apache-2.0",
    python_requires=">=3.8",
    packages=find_packages(
        where=".",
        include=["treerpo", "treerpo.*"],
        exclude=("tests", "examples"),
    ),
    install_requires=_read_requirements(),
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    include_package_data=True,
    zip_safe=False,
)
