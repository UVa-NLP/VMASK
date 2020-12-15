# VMASK
Code for the paper "Learning Variational Word Masks to Improve the Interpretability of Neural Text Classifiers"

### Requirement:
- python == 3.7.3
- gensim == 3.4.0
- pytorch == 1.2.0
- numpy == 1.16.4

### Data:
Download the [data with splits](https://drive.google.com/drive/folders/12ZbWntUFGoO2WF6Ut33WzW8ochroYFbR?usp=sharing) for CNN/LSTM-VMASK.

Download the [data with splits](https://drive.google.com/drive/folders/1ybKuzvAhyMqgpDjITVVA3USPhz6pehfq?usp=sharing) for BERT-VMASK.

Put the data under the same folder with the code.

### Train VMASK Models:

For BERT-VMASK, we adopt the BERT-base model built by huggingface: https://github.com/huggingface/transformers. We first trained BERT-base model, and then loaded its word embeddings for training BERT-VMASK. You can download our [pretrained BERT-base models](https://drive.google.com/drive/folders/19nzAv1wWtM5UCNAORMqrNtBH6j3ZXmSz?usp=sharing), and put them under the same folder with the code.

In each folder, run the following command to train VMASK-based models.
```
python main.py --save /path/to/your/model
```
Fine-tune hyperparameters (e.g. learning rate, the number of hidden units) on each dataset.

### Reference:
If you find this repository helpful, please cite our paper:
```bibtex
@inproceedings{chen2020learning,
  title={Learning Variational Word Masks to Improve the Interpretability of Neural Text Classifiers},
  author={Chen, Hanjie and Ji, Yangfeng},
  booktitle={EMNLP},
  url={https://arxiv.org/abs/2010.00667},
  year={2020}
}
```
