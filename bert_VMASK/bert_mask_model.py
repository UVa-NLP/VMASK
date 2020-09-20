import torch
import torch.nn as nn
import torch.nn.functional as F


SMALL = 1e-08


class VMASK(nn.Module):
    def __init__(self, args):
        super(VMASK, self).__init__()

        self.device = args.device
        self.mask_hidden_dim = args.mask_hidden_dim
        self.activations = {'tanh': torch.tanh, 'sigmoid': torch.sigmoid, 'relu': torch.relu,
                            'leaky_relu': F.leaky_relu}
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
            r = F.gumbel_softmax(p, hard=True, dim=2)[:, :, 1:2]
            x_prime = r * x
            return x_prime
        else:
            probs = F.softmax(p, dim=2)[:, :, 1:2]  # select the probs of being 1
            x_prime = probs * x
            return x_prime

    def get_statistics_batch(self, embeds):
        p = self.forward_sent_batch(embeds)
        return p


class MASK_BERT(nn.Module):

    def __init__(self, args, prebert):
        super(MASK_BERT, self).__init__()
        self.args = args
        self.maskmodel = VMASK(args)
        self.bertmodel = prebert

    def forward(self, inputs, flag):
        x = self.bertmodel.bert.embeddings(inputs['input_ids'], inputs['token_type_ids'])

        # Mask
        p = self.maskmodel.get_statistics_batch(x)
        x_prime = self.maskmodel(x, p, flag)

        output = self.bertmodel(x_prime, inputs['input_ids'], inputs['token_type_ids'], inputs['attention_mask'], inputs['labels'])

        probs_pos = F.softmax(p,dim=2)[:,:,1]
        probs_neg = F.softmax(p,dim=2)[:,:,0]
        self.infor_loss = torch.mean(probs_pos * torch.log(probs_pos+1e-8) + probs_neg*torch.log(probs_neg+1e-8))

        return output
