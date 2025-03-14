from setuptools import find_packages, setup

setup(
    name="icr_evaluation_pipeline",
    packages=find_packages(exclude=["icr_evaluation_pipeline_tests"]),
    install_requires=[
        "dagster",
        "dagster-cloud"
    ],
    extras_require={"dev": ["dagster-webserver", "pytest"]},
)
