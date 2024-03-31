import setuptools 
  
with open("README.md", "r") as fh: 
    description = fh.read() 
  
setuptools.setup(
    metadata_version='1.0',
    name="pyacf", 
    version="0.1", 
    author="Chengfei Wang", 
    author_email="cbw5803@psu.edu", 
    packages=["pyacf"], 
    description="This is a naive python implementation of Aggregated Channel Feature based face detection", 
    long_description=description, 
    long_description_content_type="text/markdown", 
    url="https://github.com/cbw5803/PyACF",
    license='MIT', 
    python_requires='>=3.6', 
    install_requires=[
        'numpy',
        'scikit-learn<1.2',
        'opencv-python',
        'matplotlib'
    ]
) 