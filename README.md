# GatedTabTransformer

A GatedTabTransformer implementation using Pytorch. This contain a Transformer Encoder implementation, a gMLP and the main model. Below you can see how to configure the enviroment to run the experiment and save the checkpoints.

# Table of Contents
1. [Installation](#installation)
2. [Test-the-model](#Test-the-model)
3. [Selecting-the-dataset](#selecting-the-dataset)

# Installation
First, we need to install the dependencies to run the code: 
```bash
pip install -r requirements.txt
```
IF WANT TO USE CUDA, VERIFY IF CUDA AND PYTORCH COMPATIBLE IS INSTALLED AND INSTALL ONLY THE OTHER DEPENDENCIES.

Finally, create an empty folder named “models” in the same directory as the “main.py” file.

# Test-the-model
To test the model, run the main.py file:

```bash
python3 main.py
```

# Selecting the dataset
In the terminal, you will see options to choose between the 1995 Income, Blastchar and Bank Marketing datasets. Select the desired dataset and wait for the model to train to see the AUC score. 



