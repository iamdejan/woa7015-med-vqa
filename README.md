# [WOA7015] Med-VQA Model Comparison: CNN-LSTM vs ResNet-BERT

This is the code used in the final report of WOA7015 (Advanced Machine Learning) subject, which is Med-VQA model comparison.

## Prerequisites

- Python 3.10 and Pip: needed for Jupyter to run the code. Even though newer Python versions can be used *in theory*, we used Python 3.10 as per the specification at UM DICC.
- Dependencies: Install by running `pip install -r requirements.txt`.
- Jupyter notebook: Run `pip install jupyterlab`, or follow the instruction [here](https://jupyterlab.readthedocs.io/en/stable/getting_started/installation.html) for other package managers.

## Code Structures

- `Code.ipynb`: this is the Jupyter `ipynb` file, which contains the outputs. However, there might be rendering issues.
- `utils.py`: contains common functions that will be used during training and testing phase.
- `*.py`: contains the code for training and testing the models. The codes are the same as the one in Jupyter file, only split into different files.

## How to Run Code

### Python Files

#### Training

Run these commands in sequence:
1. `python3 01_extract.py`
2. `python3 02_eda.py`
3. `python3 03_topk_coverage.py`
4. `python3 04_train_cnn_lstm.py`
5. `python3 05_train_resnet18_bert.py`
6. `python3 06_train_resnet34_bert.py`
7. `python3 07_train_resnet50_bert.py`

#### Testing

These are the test files to check the results of the closed-ended questions (yes/no answers) and open-ended questions (free-text answers).

Run these commands in sequence:
1. `python3 08_resnet50_bert_closed_ended_questions.py`
2. `python3 09_resnet50_bert_open_ended_questions.py`
3. `python3 10_resnet50_bert_sample_100_questions.py`
4. `python3 11_resnet18_bert_closed_ended_questions.py`
5. `python3 12_resnet18_bert_open_ended_questions.py`
6. `python3 13_resnet18_bert_sample_100_questions.py`

### IPYNB File

#### Terminal

Execute this command: `jupyter execute Code.ipynb`

#### Jupyter Lab

1. Start Jupyter lab.
2. Open `Code.ipynb` file from Jupyter lab.
3. If asked, pick the Python kernel.
