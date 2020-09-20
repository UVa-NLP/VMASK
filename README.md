# VMASK
Code for the paper "Learning Variational Word Masks to Improve the Interpretability of Neural Text Classifiers"

### Requirement:
- python == 3.7.3
- gensim == 3.4.0
- pytorch == 1.2.0
- numpy == 1.16.4

### Data:
Download the preprocessing [data with splits](https://drive.google.com/file/d/1n9wVSsPBjIu9Ni0GodF21nikgrSSKfWR/view?usp=sharing).

### Train VMASK Model:

For BERT-VMASK, we adopt the BERT-base model built by huggingface: https://github.com/huggingface/transformers.

In each folder, run the following command to train VMASK-based models.
```
python main.py
```

### Reference:
If you find this repository helpful, please cite our paper
