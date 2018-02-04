# CS7641 Machine Learning Assignment 1

This base code comes from the template provided by Jonathan Tay from his repository at https://github.com/JonathanTay/CS-7641-assignment-1 which has been modified to use a different dataset.

Initial testing of the base code used his datasets as shown below then modified to use a different dataset.

This file describes the structure of this assignment submission. 

## Spring 2018 

This is the code for Assignment 1 for the OMSCS CS7641 Machine Learning course taught in the Spring of 2018.

## Requirements

The assignment code was originally written in Python 3.5.1 but for this assignment it is being run in Python 3.6.0 on Microsoft Windows using the PyCharm IDE.

### Hardware and Platforms

Both are running on Windows 7 Professional 64-bit Editions.
Hardware for processing the code.
 - Lenovo Thinkpad T460P with 32Gb RAM, SSD and i7-6820HQ 8-core
 - Dell Precision T1650 with 16Gb RAM, SSD and i7-3770 8-core

The Dell desktop is slightly faster at computational processing by about 11% overall.
The Lenovo laptop is my primary development environment and follows me everywhere.

Processor and memory access speeds seems to be the gating factor for the processing time.
 - http://cpu.userbenchmark.com/Compare/Intel-Core-i7-6820HQ-vs-Intel-Core-i7-3770K/m43500vs1317

### Python Environment

Lenovo Thinkpad
- Python 3.6.0 for Windows x86-64 retrieved from https://www.python.org/downloads/windows/ Dec 2016.
- PyCharm 2017.3.3 Professional Edition IDE retrieved January 2018.
- VirtualEnv used for isolating environment from other projects
- Github account used for hosting this code and data

Dell Precision
- Python 3.5.0 for Windows x86-64 retrieved from https://www.python.org/downloads/windows/.
- PyCharm 2017.3.3 Professional Edition IDE retrieved January 2018.

Note: I did not use the Anaconda build of Python but the python.org version.

### Libraries

Library dependencies are: 
 - scikit-learn 0.19.1
 - numpy 0.14.0
 - pandas 0.22.0
 - matplotlib 2.1.2
 - tables 3.4.2
 - scipy 1.0.0

Other libraries used are part of the Python standard library. 

## Contents

The main folder contains the following files:
1. ./jtay-data adult.*, madelon_*.* ->
   These are the original datasets, as downloaded from the UCI Machine Learning Repository
 - https://archive.ics.uci.edu/ml/datasets/adult
 - https://archive.ics.uci.edu/ml/datasets/madelon
 Note: This is the original data for the base template for validation of the code.
1. ./jmm-data parkinsons.data parkinsons_updrs.data ->
   These are the original datasets, as downloaded from the UCI Machine Learning Repository
 - https://archive.ics.uci.edu/ml/datasets/parkinsons
 - https://archive.ics.uci.edu/ml/datasets/parkinsons+telemonitoring
 Note: This is the actual data for the assignment.
2. datasets.hdf -> A pre-processed/cleaned up copy of the datasets. This file is created by the parse-xxx-data.py code.
   Note: Migrate the dataset.hdf manually to the root for processing.
3. "parse-xxx-data.py" -> This python script pre-processes the original UCI ML repo files into a cleaner form for the experiments
4. "xxx-analysis.pdf" -> The analysis for this assignment.
5. helpers.py -> A collection of helper functions used for this assignment
6. ANN.py -> Code for the Neural Network Experiments
7. Boosting.py -> Code for the Boosted Tree experiments
8. "Decision Tree.py" -> Code for the Decision Tree experiments
9. KNN.py -> Code for the K-nearest Neighbours experiments
10. SVM.py -> Code for the Support Vector Machine (SVM) experiments
11. plotter.py -> Code to plot the learning and validation curves in the report
12. README.txt -> This file

## Supplemental Content

- Weka 3.8.2 for Windows x86-64 retrieved from https://www.cs.waikato.ac.nz/ml/weka/downloading.html Feb 2018.

## Outputs

There is also a subfolder called "output". This folder contains the experimental results.

Here, I use DT/ANN/BT/KNN/SVM_Lin/SVM_RBF to refer to decision trees, artificial neural networks, boosted trees, K-nearest neighbours, linear and RBF kernel SVMs respectively. A suffix of _OF indicates a deliberately "overfitted" version of the model where regularisation is turned off.

The datasets are adult/madelon referring to the two datasets used (the UCI Adult dataset and the UCI Madelon dataset)

There are 83 files in this folder. They come the following types:
1. <Algorithm>_<dataset>_reg.csv -> The validation curve tests for <algorithm> on <dataset>
2. <Algorithn>_<dataset>_LC_train.scv -> Table of # of examples vs. CV training accuracy (for 5 folds) for <algorithm> on <dataset>. For learning curves
3. <Algorithn>_<dataset>_LC_test.csv -> Table of # of examples vs. CV testing accuracy (for 5 folds) for <algorithm> on <dataset>. For learning curves
4. <Algorithm>_<dataset>_timing.csv -> Table of fraction of training set vs. training and evaluation times. If the fulll training set is of size T and a fraction f are used for training, then the evaluation set is of size (T-fT)= (1-f)T
5. ITER_base_<Algorithm>_<dataset>.csv -> Table of results for learning curves based on number of iterations/epochs.
6. ITERtestSET_<Algorithm>_<dataset>.csv -> Table showing training and test set accuracy as number of iterations/epochs is varied. NOT USED in report.
7. "test results.csv" -> Table showing the optimal hyper-parameters chosen, as well as the final accuracy on the held out test set.
8. "test results Madelon No feature selection.csv" -> Table showing the optimal hyper-parameters chosen, as well as the final accuracy on the held out test set on Madelon with feature selection turned off. (Feature selection can be turned off my removing the "Cull<X>" stages in the experiment pipelines (pipeM objects). Note that these results were done before random seeds were fixed throughout the code, so any attempt to regenerate them will be slightly different due to different random seeds.
