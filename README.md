# Multi-Class-Vehicle-Classification
A system that can detect and classify vehicles using deep learning.


<!-- TABLE OF CONTENTS -->
## Table of Contents

* [About the Project](#about-the-project)
  * [Tech Stack](#tech-stack)
  * [File Structure](#file-structure)
* [Getting Started](#getting-started)
  * [Installation](#installation)
* [Results and Demo](#results-and-demo)
* [Future Work](#future-work)
* [Troubleshooting](#troubleshooting)
* [Contributors](#contributors)
* [Acknowledgements and Resources](#acknowledgements-and-resources)


<!-- ABOUT THE PROJECT -->
## About The Project

Our Project aims to analyse a large dataset of images containing various vehicle categories. We have built a Convolutional Neural Network utilizing LeNet Architecture to detect and classify vehicles from mulitple angles. The architecture consists of two sets of convolutional and average pooling layers, followed by a flattening convoutional layer, then two fully-connected layers finally use a softmax classifier.

Refer this [documentation](https://github.com/akshayb80/Multi-Class-Vehicle-Classification/blob/master/Docs/Project%20Report.pdf)

### Tech Stack
* [OpenCV](https://opencv.org/)
* [Jupyter Notebook](https://jupyter.org/)
* [Pycharm](https://www.jetbrains.com/pycharm/)  

### File Structure
    .
    ├── docs                    # Documentation files
    │   ├── report.pdf          # Project report
    │   └── results             # Video feed of the Working Model
    ├── src
        ├── hell.model          # CNN Model
    │   ├── main.py             # Main File
    │   └── train_network.py    # Training Network
    ├── ...
    ├── test                    # Test files
    │   ├── test_network.py     # Testing
    ├── ...
    ├── LICENSE
    ├── README.md 
    ├── Setup.md                # Installation
    └── todo.md                 # Future Developments
    

<!-- GETTING STARTED -->
## Getting Started

### Prerequisites

* Anaconda Environment

  You can visit the [Anaconda](https://www.anaconda.com/) for the installation packages.

* Tensorflow-GPU version 2.1.0 (GPU version is recommended for faster performance)

  Tensorflow installation in [Conda Environment](https://docs.anaconda.com/anaconda/user-guide/tasks/tensorflow/)

  Command for One Step installation (If the system has NVIDIA GPU):

```sh
conda create --name tensor_gpu tensorflow-gpu anaconda
```

* OpenCV version 4.3.0
```sh
conda install -c conda-forge opencv
```


### Installation
1. Clone the repo
```sh
git clone https://github.com/akshayb80/Multi-Class-Vehicle-Classification.git
```

<!-- RESULTS AND DEMO -->
## Results and Demo
A video demonstrating our working model  
[**Working Model Video**](https://github.com/akshayb80/Multi-Class-Vehicle-Classification/blob/master/Docs/Multiclass%20Classification%20Test%20Video.wmv)  


<!-- FUTURE WORK -->
## Future Work
- [ ] Integrate this project with the License Plate Rocognition System


<!-- TROUBLESHOOTING -->
## Troubleshooting
* Mulitple epochs required to get the best accuracy
* Ensure there is no Tensorflow compatibilty issues with the GPU before training


<!-- CONTRIBUTORS -->
## Contributors
* [Akshay Bakshi](https://github.com/akshayb80)
* [Purvank Bhiwgade](https://github.com/purvankbhiwgade)
* [Tushar Bauskar](https://github.com/tusharb12-hash)
* [Mohak Chandani](https://github.com/MohakChandani)


<!-- ACKNOWLEDGEMENTS AND REFERENCES -->
## Acknowledgements and Resources
* [SRA VJTI](http://sra.vjti.info/) Eklavya 2020  
* Refered [OpenCV Tutorials](https://youtu.be/Z78zbnLlPUA) for Image Processing
* Refered [MIT Deep Learning](https://youtu.be/njKP3FqW3Sk) for bulding Neural Network Model
* Refered [Pyimagesearch](https://www.pyimagesearch.com) for additional OpenCV materials
...
 
