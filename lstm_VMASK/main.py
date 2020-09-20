import argparse
import torch
from torch import nn
from mask_lstm_model import *
from getvectors import getVectors
import os
import numpy as np
import random
import dill
import torch.optim as optim

os.environ["CUDA_VISIBLE_DEVICES"]="1"
parser = argparse.ArgumentParser(description='MASK_LSTM text classificer')
parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate')
parser.add_argument('-beta', type=float, default=1, help='beta')
parser.add_argument('--weight_decay', default=0, type=float, help='adding l2 regularization')
parser.add_argument('--clip', type=float, default=1, help='gradient clipping')
parser.add_argument('-epochs', type=int, default=100, help='number of epochs for training')
parser.add_argument('-batch-size', type=int, default=64, help='batch size for training')
parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')
parser.add_argument('-dropout', type=float, default=0.2, help='the probability for dropout')
parser.add_argument('-embed-dim', type=int, default=300, help='original number of embedding dimension')
parser.add_argument('-lstm-hidden-dim', type=int, default=100, help='number of hidden dimension')
parser.add_argument('-lstm-hidden-layer', type=int, default=1, help='number of hidden layers')
parser.add_argument('-mask-hidden-dim', type=int, default=300, help='number of hidden dimension')
parser.add_argument("--max_sent_len", type=int, dest="max_sent_len", default=250, help='max sentence length')
parser.add_argument("--activation", type=str, dest="activation", default="tanh", help='the choice of \
        non-linearity transfer function')
parser.add_argument('--save', type=str, default='masklstm.pt', help='path to save the final model')
parser.add_argument('--mode', type=str, default='static', help='available models: static, non-static')
parser.add_argument('--gpu', default=0, type=int, help='0:gpu, -1:cpu')
parser.add_argument('--gpu_id', default='0', type=str, help='gpu id')
parser.add_argument('--seed', type=int, default=1111, help='random seed')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)


def random_seed():
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    return


if args.gpu > -1:
    args.device = "cuda"
else:
    args.device = "cpu"


# load data
data = dill.load(open("data.pkl", "rb"))
train_text = data.train_text
train_label = data.train_label
dev_text = data.dev_text
dev_label = data.dev_label
test_text = data.test_text
test_label = data.test_label

wordvocab = data.wordvocab

vectors = getVectors(args, wordvocab)


args.embed_num = len(wordvocab)
args.class_num = 2


class B:
    text = torch.zeros(1).to(args.device)
    label = torch.zeros(1).to(args.device)


def batch_from_list(textlist, labellist):
    batch = B()
    batch.text = textlist[0]
    batch.label = labellist[0]
    for txt, la in zip(textlist[1:], labellist[1:]):
        batch.text = torch.cat((batch.text, txt), 0)
        # you may need to change the type of "la" to torch.tensor for different datasets, sorry for the inconvenience
        batch.label = torch.cat((batch.label, la), 0) # for SST and IMDB dataset, you do not need to change "la" type
    batch.text = batch.text.to(args.device)
    batch.label = batch.label.to(args.device)
    return batch


# evaluate
def evaluation(model, data_text, data_label):
    model.eval()
    acc, loss, size = 0, 0, 0
    count = 0
    for stidx in range(0, len(data_label), args.batch_size):
        count += 1
        batch = batch_from_list(data_text[stidx:stidx + args.batch_size],
                                data_label[stidx:stidx + args.batch_size])
        pred = model(batch, 'eval')

        batch_loss = criterion(pred, batch.label)
        loss += batch_loss.item()

        _, pred = pred.max(dim=1)
        acc += (pred == batch.label).sum().float()
        size += len(pred)

    acc /= size
    loss /= count
    return loss, acc


def main():
    # load model
    model = MASK_LSTM(args, vectors)
    model.to(torch.device(args.device))
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model.train()
    best_val_acc = None
    beta = args.beta
    for epoch in range(1, args.epochs+1):
        model.train()
        print("\n## The {} Epoch, All {} Epochs ! ##".format(epoch, args.epochs))
        lstm_count = 0
        trn_lstm_size, trn_lstm_corrects, trn_lstm_loss = 0, 0, 0

        # shuffle
        textlist1 = train_text.copy()
        labellist1 = train_label.copy()
        listpack = list(zip(textlist1, labellist1))
        random.shuffle(listpack)
        textlist1[:], labellist1[:] = zip(*listpack)

        for stidx in range(0, len(labellist1), args.batch_size):
            lstm_count += 1
            batch = batch_from_list(textlist1[stidx:stidx + args.batch_size],
                                    labellist1[stidx:stidx + args.batch_size])
            pred = model(batch, 'train')
            optimizer.zero_grad()
            model_loss = criterion(pred, batch.label)
            batch_loss = model_loss + beta * model.infor_loss
            trn_lstm_loss += batch_loss.item()
            batch_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip)
            optimizer.step()

            _, pred = pred.max(dim=1)
            trn_lstm_corrects += (pred == batch.label).sum().float()
            trn_lstm_size += len(pred)

        dev_lstm_loss, dev_lstm_acc = evaluation(model, dev_text.copy(), dev_label.copy())
        if not best_val_acc or dev_lstm_acc > best_val_acc:
            with open(args.save, 'wb') as f:
                torch.save(model, f)
            best_val_acc = dev_lstm_acc

        train_lstm_acc = trn_lstm_corrects / trn_lstm_size
        train_lstm_loss = trn_lstm_loss / lstm_count
        print('local_epoch {} | train_lstm_loss {:.6f} | train_lstm_acc {:.6f} | dev_lstm_loss {:.6f} | '
              'dev_lstm_acc {:.6f} | best_dev_acc {:.6f}'.format(epoch, train_lstm_loss, train_lstm_acc,
                                                                 dev_lstm_loss, dev_lstm_acc, best_val_acc))

        # annealing
        if epoch % 10 == 0:
            if beta > 0.01:
                beta -= 0.099


    # load best model and test
    del model
    with open(args.save, 'rb') as f:
        model = torch.load(f)
    model.to(torch.device(args.device))
    _, test_acc = evaluation(model, test_text.copy(), test_label.copy())
    print('\nfinal_test_acc {:.6f}'.format(test_acc))


if __name__ == "__main__":
    random_seed()
    criterion = nn.CrossEntropyLoss()
    main()
