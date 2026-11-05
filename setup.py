from setuptools import setup, find_packages

# Read dependencies from requirements.txt
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="open-unlearning",
    version="0.1.0",
    author="Vineeth Dorna, Anmol Mekala",
    author_email="vineethdorna@gmail.com, m.anmolreddy@gmail.com",
    description="A library for machine unlearning in LLMs.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/locuslab/open-unlearning",
    license="MIT",
    packages=find_packages(),
    install_requires=requirements,  # Uses requirements.txt
    extras_require={
        "lm-eval": [
            "lm-eval==0.4.8",
        ],  # Install using `pip install ".[lm-eval]"`
        "paraphrase-judge": [
            "openai>=1.0.0",
            "google-genai>=1.0.0",
            "nltk==3.9.2",
        ],  # LLM-judge + repetitiveness deps for the paraphrase evals. Install using `pip install ".[paraphrase-judge]"`
        "hpsearch": [
            "optuna==4.7.0",
        ],  # Bayesian hyperparameter search scripts (scripts/hpsearch_atwu.py). Install using `pip install ".[hpsearch]"`
        "dev": [
            "pre-commit==4.0.1",
            "ruff==0.6.9",
        ],  # Install using `pip install ".[dev]"`
    },
    python_requires=">=3.11",
)
