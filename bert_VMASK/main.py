from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import random

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from pytorch_transformers import (WEIGHTS_NAME, BertConfig,
                                  BertForSequenceClassification, BertTokenizer)

from pytorch_transformers import AdamW, WarmupLinearSchedule

from utils_glue import (convert_examples_to_features,
                        output_modes, processors)

from bert_mask_model import *


os.environ["CUDA_VISIBLE_DEVICES"]="1"

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer)
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    set_seed(args)
    epochnum = 0
    best_val_acc = None
    beta = args.beta
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        logging_steps = 0
        preds = None
        out_label_ids = None
        epochnum += 1
        count, trn_model_loss = 0, 0
        for step, batch in enumerate(epoch_iterator):
            count += 1
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,  # XLM and RoBERTa don't use segment_ids
                      'labels':         batch[3]}
            outputs = model(inputs, 'train')
            model_loss, logits = outputs[:2]

            batch_loss = model_loss + beta * model.infor_loss
            trn_model_loss += batch_loss.item()

            if args.n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()
            global_step += 1
            logging_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

        preds = np.argmax(preds, axis=1)
        tr_acc = (preds == out_label_ids).mean()

        # evaluate model
        eval_dataset = load_and_cache_examples(args, args.task_name, tokenizer, type='dev')
        eval_loss, eval_acc = evaluate(args, model, eval_dataset)
        if not best_val_acc or eval_acc > best_val_acc:
            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir)
            with open(os.path.join(args.output_dir, args.savename), 'wb') as f:
                torch.save(model, f)
            tokenizer.save_pretrained(args.output_dir)
            torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))
            best_val_acc = eval_acc

        tr_loss = trn_model_loss / count
        print('epoch {} | train_loss {:.6f} | train_acc {:.6f} | dev_loss {:.6f} | dev_acc {:.6f}'.format(epochnum,
                                                                                                          tr_loss,
                                                                                                          tr_acc,
                                                                                                          eval_loss,
                                                                                                          eval_acc))

        if epochnum % 1 == 0:
           if beta > 0.01:
               beta -= 0.099

    return global_step, tr_loss


def evaluate(args, model, eval_dataset):
    eval_output_dir = args.output_dir
    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,  # XLM and RoBERTa don't use segment_ids
                      'labels':         batch[3]}
            outputs = model(inputs, 'eval')
            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs['labels'].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    preds = np.argmax(preds, axis=1)
    eval_acc = (preds == out_label_ids).mean()

    return eval_loss, eval_acc


def load_and_cache_examples(args, task, tokenizer, type):
    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}'.format(
        type,
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length),
        str(task)))
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if type == 'train':
            examples = processor.get_train_examples(args.data_dir)
        elif type == 'dev':
            examples = processor.get_dev_examples(args.data_dir)
        else:
            examples = processor.get_test_examples(args.data_dir)
        features = convert_examples_to_features(examples, label_list, args.max_seq_length, tokenizer, output_mode,
                                                cls_token_at_end=bool(args.model_type in ['xlnet']),
                                                # xlnet has a cls token at the end
                                                cls_token=tokenizer.cls_token,
                                                cls_token_segment_id=2 if args.model_type in ['xlnet'] else 0,
                                                sep_token=tokenizer.sep_token,
                                                sep_token_extra=bool(args.model_type in ['roberta']),
                                                # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                                                pad_on_left=bool(args.model_type in ['xlnet']),
                                                # pad on the left for xlnet
                                                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
                                                )
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    return dataset


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default='./dataset/data_name', type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_type", default='bert', type=str,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default='bert-base-uncased', type=str,
                        help="Path to pre-trained model or shortcut name selected in the list: ")
    parser.add_argument("--task_name", default='SST-2', type=str,
                        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()))
    parser.add_argument("--output_dir", default='./output', type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument('--savename', type=str, default='maskbert.pt',
                        help='path to save the final model')

    ## Other parameters
    parser.add_argument('-beta', type=float, default=1, help='beta')
    parser.add_argument('-mask-hidden-dim', type=int, default=100, help='number of hidden dimension')
    parser.add_argument("--activation", type=str, dest="activation", default="tanh", help='the choice of \
            non-linearity transfer function')
    parser.add_argument('-embed-dim', type=int, default=768, help='original number of embedding dimension')
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=25, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_lower_case", default=True,
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=1e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=10.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=10.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--gpu', default=0, type=int, help='0:gpu, -1:cpu')
    parser.add_argument('--gpu_id', default='0', type=str, help='gpu id')
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    if args.gpu > -1:
        args.device = "cuda"
    else:
        args.device = "cpu"
    args.n_gpu = 1

    # Set seed
    set_seed(args)

    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)
    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    logger.info("Training/evaluation parameters %s", args)

    # Load pretrained bert model
    prebert = model_class.from_pretrained('./output_base/pre_model_name')
    tokenizer = tokenizer_class.from_pretrained('./output_base/pre_model_name', do_lower_case=args.do_lower_case)
    prebert.to(args.device)

    model = MASK_BERT(args, prebert)
    model.to(args.device)

    # fix embeddings
    parameters = filter(lambda p: p.requires_grad, model.bertmodel.bert.embeddings.parameters())
    for param in parameters:
        param.requires_grad = False

    # Training
    train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, type='train')
    global_step, tr_loss = train(args, train_dataset, model, tokenizer)
    logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Load the well-trained model and vocabulary that you have fine-tuned
    del model
    # model = model_class.from_pretrained(args.output_dir)
    with open(os.path.join(args.output_dir, args.savename), 'rb') as f:
        model = torch.load(f)
    model.to(torch.device(args.device))
    tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)

    # Test
    test_dataset = load_and_cache_examples(args, args.task_name, tokenizer, type='test')
    test_loss, test_acc = evaluate(args, model, test_dataset)
    print('\ntest_loss {:.6f} | test_acc {:.6f}'.format(test_loss, test_acc))
    return test_loss, test_acc


if __name__ == "__main__":
    main()
