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

### Tech Stack
* [OpenCV](https://opencv.org/)
* [Jupyter Notebook](https://jupyter.org/)
* [Pycharm](https://www.jetbrains.com/pycharm/)  

### File Structure
    .
    ├── docs                    # Documentation files (alternatively `doc`)
    │   ├── report.pdf          # Project report
    │   └── results             # Folder containing screenshots, gifs, videos of results
    ├── src
        ├── hell.model          # Load and stress tests
    │   ├── main.py             # End-to-end, integration tests (alternatively `e2e`)
    │   └── train_network.py    # Source files
    ├── ...
    ├── test                    # Test files (alternatively `spec` or `tests`)
    │   ├── test_network.py     # Load and stress tests
    ├── ...
    ├── LICENSE
    ├── README.md 
    ├── Setup.md                # If Installation instructions are lengthy
    └── todo.md                 # If Future developments and current status gets lengthy
    

<!-- GETTING STARTED -->
### Installation
1. Clone the repo
```sh
git clone https://github.com/your_username_/Project-Name.git
```

<!-- RESULTS AND DEMO -->
## Results and Demo
A video demonstrating our working model  
[**result gif or video**](https://result.gif)  


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
 
