import torch
import torch.nn as nn
import torch.nn.functional as F


SMALL = 1e-08


class VMASK(nn.Module):
	def __init__(self, args):
		super(VMASK, self).__init__()

		self.device = args.device
		self.mask_hidden_dim = args.mask_hidden_dim
		self.activations = {'tanh': torch.tanh, 'sigmoid': torch.sigmoid, 'relu': torch.relu}
		self.activation = self.activations[args.activation]
		self.embed_dim = args.embed_dim
		self.linear_layer = nn.Linear(self.embed_dim, self.mask_hidden_dim)
		self.hidden2p = nn.Linear(self.mask_hidden_dim, 2)
		
	def forward_sent_batch(self, embeds):

		temps = self.activation(self.linear_layer(embeds))
		p = self.hidden2p(temps)  # bsz, seqlen, dim
		return p

	def forward(self, x, p, flag):
		if flag == 'train':
			r = F.gumbel_softmax(p,hard=True,dim=2)[:,:,1:2]
			x_prime = r * x
			return x_prime
		else:
			probs = F.softmax(p,dim=2)[:,:,1:2]
			x_prime = x * probs
			return x_prime

	def get_statistics_batch(self, embeds):
		p = self.forward_sent_batch(embeds)
		return p


class CNN(nn.Module):
	def __init__(self, args, vectors):
		super(CNN, self).__init__()

		self.args = args

		self.word_emb = nn.Embedding(args.embed_num, args.embed_dim, padding_idx=1)
		# initialize word embedding with pretrained word2vec
		if args.mode != 'rand':
			# pass
			self.word_emb.weight.data.copy_(torch.from_numpy(vectors))
		if args.mode in ('static', 'multichannel'):
			self.word_emb.weight.requires_grad = False
		if args.mode == 'multichannel':
			self.word_emb_multi = nn.Embedding(args.embed_num, args.embed_dim, padding_idx=1)
			self.word_emb_multi.weight.data.copy_(torch.from_numpy(vectors))
			self.in_channels = 2
		else:
			self.in_channels = 1

		# <unk> vector is randomly initialized
		nn.init.uniform_(self.word_emb.weight.data[0], -0.01, 0.01)

		# <pad> vector is initialized as zero padding
		nn.init.constant_(self.word_emb.weight.data[1], 0)

		for filter_size in args.kernel_sizes:
			conv = nn.Conv1d(self.in_channels, args.kernel_num, args.embed_dim * filter_size, stride=args.embed_dim)
			setattr(self, 'conv_' + str(filter_size), conv)

		self.fc = nn.Linear(len(args.kernel_sizes) * args.kernel_num, args.class_num)

	def forward(self, x):
		batch_size, seq_len, _ = x.shape
		conv_in = x.view(batch_size, 1, -1)
		conv_result = [
			F.max_pool1d(F.relu(getattr(self, 'conv_' + str(filter_size))(conv_in)), seq_len - filter_size + 1).view(-1,
																													 self.args.kernel_num)
			for filter_size in self.args.kernel_sizes]

		R_out = torch.cat(conv_result, 1)
		out = F.dropout(R_out, p=self.args.dropout, training=self.training)
		out = self.fc(out)

		return out


class VMASK_CNN(nn.Module):
	def __init__(self, args, vectors):
		super(VMASK_CNN, self).__init__()
		self.args = args
		self.embed_dim = args.embed_dim
		self.device = args.device
		self.sample_size = args.sample_size
		self.max_sent_len = args.max_sent_len

		self.vmask = VMASK(args)
		self.cnnmodel = CNN(args, vectors)

	def forward(self, batch, flag):
		# embedding
		x = batch.text
		x = self.cnnmodel.word_emb(x)
		# MASK
		p = self.vmask.get_statistics_batch(x)
		x_prime = self.vmask(x, p, flag)
		output = self.cnnmodel(x_prime)

		probs_pos = F.softmax(p,dim=2)[:,:,1]
		probs_neg = F.softmax(p,dim=2)[:,:,0]
		self.infor_loss = torch.mean(probs_pos * torch.log(probs_pos+1e-8) + probs_neg*torch.log(probs_neg+1e-8))

		return output
