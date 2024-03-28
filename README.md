# SHREC2024 - Hand Motion Recognition  

## Description

This repository contains the submission, i.e. the code and models, by
* Martin Hanik,
* Esfandiar Navayazdani, and
* Christoph von Tycowicz

to the [Recognition Of Hand Motions Molding Clay](https://www.shrec.net/SHREC-2024-hand-motion/) track of the 3D Shape Retrieval Challenge (SHREC) 2024.
The task is to classify highly similar hand motion sequences from a professional potter into one out of seven classes.
The sequences are of variable length and comprise 3D coordinate data of landmarks on both potters hands.



## Running the code

You can start a live demo of the prediction on binder:<br>
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/morphomatics/SHREC24/HEAD)


To run the test script on all test sequences execute the following
```bash
python test.py --path='./data/Test-set/**/*.txt'
```

The code that generates an ensemble based on random training/validation splits can be found in `train.py`.


## Data

The data is provided by the organizers of the SHREC 2024 challenge and can be downloaded [here](https://www.shrec.net/SHREC-2024-hand-motion/Data/Data%20Split.rar).
To set up the data, you can run the `start` script (this will happen automatically when starting the binder instance; see above)
```bash
./start
```
Afterward, the data is provided in the `data` folder. The data is split into a training and a test set.
The training set contains 50 sequences, the test set 12 sequences.
Each sequence is stored in a separate text file.
The text files contain the 3D coordinates of the landmarks of both hands.