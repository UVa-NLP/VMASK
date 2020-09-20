import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.init as init


SMALL = 1e-08


class VMASK(nn.Module):
	def __init__(self, args):
		super(VMASK, self).__init__()

		self.device = args.device
		self.mask_hidden_dim = args.mask_hidden_dim
		self.activations = {'tanh': torch.tanh, 'sigmoid': torch.sigmoid, 'relu': torch.relu, 'leaky_relu': F.leaky_relu}
		self.activation = self.activations[args.activation]
		self.embed_dim = args.embed_dim
		self.linear_layer = nn.Linear(self.embed_dim, self.mask_hidden_dim)
		self.hidden2p = nn.Linear(self.mask_hidden_dim, 2)

	def forward_sent_batch(self, embeds):

		temps = self.activation(self.linear_layer(embeds))
		p = self.hidden2p(temps)  # seqlen, bsz, dim
		return p

	def forward(self, x, p, flag):
		if flag == 'train':
			r = F.gumbel_softmax(p,hard=True,dim=2)[:,:,1:2]
			x_prime = r * x
			return x_prime
		else:
			probs = F.softmax(p,dim=2)[:,:,1:2] #select the probs of being 1
			x_prime = probs * x
			return x_prime

	def get_statistics_batch(self, embeds):
		p = self.forward_sent_batch(embeds)
		return p


class LSTM(nn.Module):
	def __init__(self, args, vectors):
		super(LSTM, self).__init__()

		self.args = args

		self.embed = nn.Embedding(args.embed_num, args.embed_dim, padding_idx=1)

		# initialize word embedding with pretrained word2vec
		self.embed.weight.data.copy_(torch.from_numpy(vectors))

		# fix embedding
		if args.mode == 'static':
			self.embed.weight.requires_grad = False
		else:
			self.embed.weight.requires_grad = True

		# <unk> vectors is randomly initialized
		nn.init.uniform_(self.embed.weight.data[0], -0.05, 0.05)

		# <pad> vector is initialized as zero padding
		nn.init.constant_(self.embed.weight.data[1], 0)

		# lstm
		self.lstm = nn.LSTM(args.embed_dim, args.lstm_hidden_dim, num_layers=args.lstm_hidden_layer)
		# initial weight
		init.xavier_normal_(self.lstm.all_weights[0][0], gain=np.sqrt(6.0))
		init.xavier_normal_(self.lstm.all_weights[0][1], gain=np.sqrt(6.0))

		# linear
		self.hidden2label = nn.Linear(args.lstm_hidden_dim, args.class_num)
		# dropout
		self.dropout = nn.Dropout(args.dropout)
		self.dropout_embed = nn.Dropout(args.dropout)

	def forward(self, x):
		# lstm
		lstm_out, _ = self.lstm(x)
		lstm_out = torch.transpose(lstm_out, 0, 1)
		lstm_out = torch.transpose(lstm_out, 1, 2)
		# pooling
		lstm_out = torch.tanh(lstm_out)
		lstm_out = F.max_pool1d(lstm_out, lstm_out.size(2)).squeeze(2)
		lstm_out = torch.tanh(lstm_out)
		lstm_out = F.dropout(lstm_out, p=self.args.dropout, training=self.training)
		# linear
		logit = self.hidden2label(lstm_out)
		out = F.softmax(logit, 1)
		return out


class MASK_LSTM(nn.Module):

	def __init__(self, args, vectors):
		super(MASK_LSTM, self).__init__()
		self.args = args
		self.embed_dim = args.embed_dim
		self.device = args.device
		self.sample_size = args.sample_size
		self.max_sent_len = args.max_sent_len

		self.vmask = VMASK(args)
		self.lstmmodel = LSTM(args, vectors)

	def forward(self, batch, flag, topk):
		# embedding
		x = batch.text.t()
		embed = self.lstmmodel.embed(x)
		embed = F.dropout(embed, p=self.args.dropout, training=self.training)
		x = embed.view(len(x), embed.size(1), -1)  # seqlen, bsz, embed-dim
		# MASK
		p = self.vmask.get_statistics_batch(x)
		x_prime = self.vmask(x, p, flag)
		output = self.lstmmodel(x_prime)

		# self.infor_loss = F.softmax(p,dim=2)[:,:,1:2].mean()
		probs_pos = F.softmax(p,dim=2)[:,:,1]
		probs_neg = F.softmax(p,dim=2)[:,:,0]
		self.infor_loss = torch.mean(probs_pos * torch.log(probs_pos+1e-8) + probs_neg*torch.log(probs_neg+1e-8))

		return output
