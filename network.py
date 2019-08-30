import torch
import numpy as np
import random
import math


def cuda_(var, cfg):
    return var.cuda() if cfg.cuda else var


def toss_(p):
    return random.randint(0, 99) <= p


def nan(v):
    if type(v) is float:
        return v == float('nan')
    return np.isnan(np.sum(v.data.cpu().numpy()))


def init_gru(gru):
    gru.reset_parameters()
    for _, hh, _, _ in gru.all_weights:
        for i in range(0, hh.size(0), gru.hidden_size):
            torch.nn.init.orthogonal_(hh[i:i + gru.hidden_size], gain=1)


class Attn(torch.nn.Module):
    def __init__(self, hidden_size):
        super(Attn, self).__init__()
        self.hidden_size = hidden_size
        self.attn = torch.nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = torch.nn.Parameter(torch.zeros(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.normal_(mean=0, std=stdv)

    def forward(self, hidden, encoder_outputs, normalize=True):
        encoder_outputs = encoder_outputs.transpose(0, 1)  # [B,T,H]
        attn_energies = self.score(hidden, encoder_outputs)
        normalized_energy = torch.nn.functional.softmax(attn_energies, dim=2)  # [B,1,T]
        context = torch.bmm(normalized_energy, encoder_outputs)  # [B,1,H]
        return context.transpose(0, 1)  # [1,B,H]

    def score(self, hidden, encoder_outputs):
        max_len = encoder_outputs.size(1)
        H = hidden.repeat(max_len, 1, 1).transpose(0, 1)
        energy = torch.tanh(self.attn(torch.cat([H, encoder_outputs], 2)))  # [B,T,2H]->[B,T,H]
        energy = energy.transpose(2, 1)  # [B,H,T]
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)  # [B,1,H]
        energy = torch.bmm(v, energy)  # [B,1,T]
        return energy


class SimpleDynamicEncoder(torch.nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, n_layers, dropout, cfg):
        super(SimpleDynamicEncoder, self).__init__()
        self.cfg = cfg
        self.hidden_size = hidden_size
        self.embedding = torch.nn.Embedding(vocab_size, embed_size)
        self.gru = torch.nn.GRU(embed_size, hidden_size, n_layers, dropout=dropout, bidirectional=True)
        init_gru(self.gru)

    def forward(self, input_seqs, input_lens, hidden=None):
        """
        forward procedure. No need for inputs to be sorted
        :param input_seqs: Variable of [T,B]
        :param hidden:
        :param input_lens: *numpy array* of len for each input sequence
        :return:
        """
        embedded = self.embedding(input_seqs)
        embedded = embedded.transpose(0, 1)  # [B,T,E]
        sort_idx = np.argsort(-input_lens)
        unsort_idx = cuda_(torch.LongTensor(np.argsort(sort_idx)), self.cfg)
        input_lens = input_lens[sort_idx]
        sort_idx = cuda_(torch.LongTensor(sort_idx), self.cfg)
        embedded = embedded[sort_idx].transpose(0, 1)  # [T,B,E]
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lens)
        outputs, hidden = self.gru(packed, hidden)
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
        outputs = outputs.transpose(0, 1)[unsort_idx].transpose(0, 1).contiguous()
        hidden = hidden.transpose(0, 1)[unsort_idx].transpose(0, 1).contiguous()
        return outputs, hidden, embedded


class CAS_Decoder(torch.nn.Module):
    def __init__(self, hidden_size, degree_size, continue_size, act_size, slot_size, dropout_rate, cfg):
        super(CAS_Decoder, self).__init__()
        self.cfg = cfg
        self.continue_in_proj = torch.nn.Linear(hidden_size + continue_size + act_size + slot_size, hidden_size)
        self.continue_attn_u = Attn(hidden_size)
        self.continue_attn_a = Attn(hidden_size)
        self.continue_attn_user_inform = Attn(hidden_size)
        self.continue_attn_user_request = Attn(hidden_size)
        self.continue_attn_agent_request = Attn(hidden_size)
        self.continue_attn_agent_propose = Attn(hidden_size)
        self.continue_out_proj = torch.nn.Linear(7*hidden_size, continue_size)

        self.act_in_proj = torch.nn.Linear(continue_size + act_size + slot_size, hidden_size)
        self.act_attn_u = Attn(hidden_size)
        self.act_attn_a = Attn(hidden_size)
        self.act_attn_user_inform = Attn(hidden_size)
        self.act_attn_user_request = Attn(hidden_size)
        self.act_attn_agent_request = Attn(hidden_size)
        self.act_attn_agent_propose = Attn(hidden_size)
        self.act_out_proj = torch.nn.Linear(7*hidden_size, act_size)

        self.slot_in_proj = torch.nn.Linear(continue_size + 2*act_size + slot_size, hidden_size)
        self.slot_attn_u = Attn(hidden_size)
        self.slot_attn_a = Attn(hidden_size)
        self.slot_attn_user_inform = Attn(hidden_size)
        self.slot_attn_user_request = Attn(hidden_size)
        self.slot_attn_agent_request = Attn(hidden_size)
        self.slot_attn_agent_propose = Attn(hidden_size)
        self.slot_out_proj = torch.nn.Linear(7*hidden_size, slot_size)

        self.continue_gru = torch.nn.GRU(7*hidden_size+degree_size, hidden_size, 1,
                                         dropout=dropout_rate, bidirectional=False)
        self.act_gru = torch.nn.GRU(7 * hidden_size + degree_size , hidden_size, 1,
                                    dropout=dropout_rate, bidirectional=False)
        self.slot_gru = torch.nn.GRU(7 * hidden_size + degree_size, hidden_size, 1,
                                     dropout=dropout_rate, bidirectional=False)
        init_gru(self.continue_gru)#lstm learning to count
        init_gru(self.act_gru)  # lstm learning to count
        init_gru(self.slot_gru)  # lstm learning to count

        self.activation = torch.nn.Tanh()

        self.dropout_rate = dropout_rate

    def forward(self, last_agent_enc_out, current_user_request_enc_out, current_user_inform_enc_out, \
                current_agent_request_enc_out, current_agent_propose_enc_out, user_enc_out, query_last_hidden,
                kb_turn, last_continue, last_act, last_slot, cas_last_hidden, current_continue=None, current_act=None, mode='test'):
        #last_contiue = [0,0]
        #last_act = [0,0,1,0,0,0,0]
        #last_slot = [0, 1, 0, 1, 0, 0, ... ]
        #use out_proj to encode last CAS
        #continue module
        continue_in = self.activation(self.continue_in_proj(torch.cat([query_last_hidden, last_continue, last_act, last_slot], dim=2)))
        continue_attn_a = self.continue_attn_a(continue_in, last_agent_enc_out)
        continue_attn_user_inform = self.continue_attn_user_inform(continue_in, current_user_inform_enc_out)
        continue_attn_user_request = self.continue_attn_user_request(continue_in, current_user_request_enc_out)
        continue_attn_agent_propose = self.continue_attn_agent_propose(continue_in, current_agent_propose_enc_out)
        continue_attn_agent_request = self.continue_attn_agent_request(continue_in, current_agent_request_enc_out)
        continue_attn_u = self.continue_attn_u(continue_in, user_enc_out)
        continue_gru_out, continue_hidden = self.continue_gru(
            torch.cat([kb_turn, continue_in, continue_attn_a, continue_attn_user_inform, continue_attn_user_request,
                       continue_attn_agent_request, continue_attn_agent_propose, continue_attn_u], dim=2),
            cas_last_hidden,
        )
        continue_out = self.continue_out_proj(torch.cat([continue_gru_out, continue_attn_a, continue_attn_user_inform, continue_attn_user_request,
                       continue_attn_agent_request, continue_attn_agent_propose, continue_attn_u], dim=2))#binary classification
        continue_out = torch.nn.functional.softmax(continue_out, dim=2)

        #act module
        if mode == 'train':
            act_in = self.activation(self.act_in_proj(torch.cat([current_continue, last_act, last_slot], dim=2)))
        else:
            act_in = self.activation(self.act_in_proj(torch.cat([continue_out, last_act, last_slot], dim=2)))
        act_attn_a = self.act_attn_a(act_in, last_agent_enc_out)
        act_attn_user_inform = self.act_attn_user_inform(act_in, current_user_inform_enc_out)
        act_attn_user_request = self.act_attn_user_request(act_in, current_user_request_enc_out)
        act_attn_agent_propose = self.act_attn_agent_propose(act_in, current_agent_propose_enc_out)
        act_attn_agent_request = self.act_attn_agent_request(act_in, current_agent_request_enc_out)
        act_attn_u = self.act_attn_u(act_in, user_enc_out)
        act_gru_out, act_hidden = self.act_gru(
            torch.cat([kb_turn, act_in, act_attn_a, act_attn_user_inform, act_attn_user_request,\
                       act_attn_agent_propose, act_attn_agent_request, act_attn_u], dim=2),
            continue_hidden,
        )
        act_out = self.act_out_proj(torch.cat([act_gru_out, act_attn_a, act_attn_user_inform, act_attn_user_request,\
                       act_attn_agent_propose, act_attn_agent_request, act_attn_u], dim=2))
        act_out = torch.nn.functional.softmax(act_out, dim=2)

        #slot module
        if mode == 'train':
            slot_in = self.activation(self.slot_in_proj(torch.cat([current_continue, current_act, last_act, last_slot], dim=2)))
        else:
            slot_in = self.activation(self.slot_in_proj(torch.cat([continue_out, act_out, last_act, last_slot], dim=2)))
        slot_attn_a = self.slot_attn_a(slot_in, last_agent_enc_out)
        slot_attn_user_inform = self.slot_attn_user_inform(slot_in, current_user_inform_enc_out)
        slot_attn_user_request = self.slot_attn_user_request(slot_in, current_user_request_enc_out)
        slot_attn_agent_propose = self.slot_attn_agent_propose(slot_in, current_agent_propose_enc_out)
        slot_attn_agent_request = self.slot_attn_agent_request(slot_in, current_agent_request_enc_out)
        slot_attn_u = self.slot_attn_u(slot_in, user_enc_out)
        slot_gru_out, slot_hidden = self.slot_gru(
            torch.cat([kb_turn, slot_in, slot_attn_a, slot_attn_user_inform, slot_attn_user_request,\
                       slot_attn_agent_propose, slot_attn_agent_request, slot_attn_u], dim=2),
            act_hidden,
        )
        slot_out = self.slot_out_proj(torch.cat([slot_gru_out, slot_attn_a, slot_attn_user_inform, slot_attn_user_request,\
                       slot_attn_agent_propose, slot_attn_agent_request, slot_attn_u], dim=2))
        slot_out = torch.sigmoid(slot_out)

        return continue_out, act_out, slot_out, slot_hidden


class Seq2CAS(torch.nn.Module):
    def __init__(self, cfg, vocab):
        super(Seq2CAS, self).__init__()
        self.vocab = vocab
        self.cfg = cfg
        self.encoder = SimpleDynamicEncoder(len(vocab), cfg.emb_size, cfg.hidden_size, cfg.encoder_layer_num, cfg.dropout_rate, cfg)
        self.decoder = CAS_Decoder(cfg.hidden_size, cfg.kb_turn_size, cfg.continue_size, cfg.act_size, cfg.slot_size, cfg.dropout_rate, cfg)
        self.max_ts = cfg.cas_max_ts
        self.continue_loss = torch.nn.NLLLoss(ignore_index=0, reduction='mean')
        self.act_loss = torch.nn.NLLLoss(ignore_index=0, reduction='mean')
        self.slot_loss = torch.nn.BCELoss(reduction='none')

    def forward(self, x, gt_y, mode, **kwargs):
        if mode == 'train' or mode == 'valid':
            pred_y = self.forward_turn(x, gt_y, mode, **kwargs)
            y = kwargs['cas_continue']
            continue_mask = y[:, :, 2]#continue dimension
            continue_mask = continue_mask.unsqueeze(2)  # 1, batchsize, 1
            seq_len = gt_y[2].size(0)
            batch_size = gt_y[2].size(1)
            slot_size = gt_y[2].size(2)
            continue_mask = continue_mask.expand((seq_len, batch_size, slot_size))
            loss = self.supervised_loss(pred_y, gt_y, continue_mask)
            return loss
        elif mode == 'test':
            pred_y = self.forward_turn(x, gt_y, mode, **kwargs)
            return pred_y

    def forward_turn(self, user, agent, mode, **kwargs):
        user_len = kwargs['user_len']#batchsize
        last_agent = kwargs['last_agent']#seqlen, batchsize
        last_agent_len = kwargs['last_agent_len']#batchsize
        current_slot = kwargs['current_slot']#seqlen, batchsize
        current_slot_len = kwargs['current_slot_len']#batchsize
        current_user_request = kwargs['current_user_request']#seqlen, batchsize
        current_user_request_len = kwargs['current_user_request_len']#batchsize
        current_user_inform = kwargs['current_user_inform']#seqlen, batchsize
        current_user_inform_len = kwargs['current_user_inform_len']#batchsize
        current_agent_request = kwargs['current_agent_request']#seqlen, batchsize
        current_agent_request_len = kwargs['current_agent_request_len']#batchsize
        current_agent_propose = kwargs['current_agent_propose']  # seqlen, batchsize
        current_agent_propose_len = kwargs['current_agent_propose_len']  # batchsize
        kb_turn = kwargs['kb_turn']# 1, batchsize, kb_turn_size
        cas_act = kwargs['cas_act']
        cas_continue = kwargs['cas_continue']
        continue_tm1 = kwargs['continue_go'] # GO token
        act_tm1 = kwargs['act_go']  # GO token
        slot_tm1 = kwargs['slot_go'] # GO token
        last_agent_enc_out, last_agent_hidden, _ = self.encoder(last_agent, last_agent_len)
        current_user_request_enc_out, current_user_request_hidden, _ = self.encoder(current_user_request, current_user_request_len, last_agent_hidden)
        current_user_inform_enc_out, current_user_inform_hidden, _ = self.encoder(current_user_inform, current_user_inform_len, current_user_request_hidden)
        current_agent_request_enc_out, current_agent_request_hidden, _ = self.encoder(current_agent_request, current_agent_request_len, current_user_inform_hidden)
        current_agent_propose_enc_out, current_agent_propose_hidden, _ = self.encoder(current_agent_propose, current_agent_propose_len, current_agent_request_hidden)
        user_enc_out, user_enc_hidden, _ = self.encoder(user, user_len, current_agent_propose_hidden)
        user_last_hidden = user_enc_hidden[:-1]
        last_hidden = user_last_hidden
        agent_length = agent[0].size(0)
        agent_continue_proba = []
        agent_act_proba = []
        agent_slot_proba = []
        if mode == 'train':
            for t in range(agent_length):
                m = 'train'
                if toss_(self.cfg.teacher_force):
                    m ='test'
                continue_proba, act_proba, slot_proba, last_hidden = self.decoder(last_agent_enc_out,
                                                            current_user_request_enc_out, \
                                                            current_user_inform_enc_out, \
                                                            current_agent_request_enc_out,\
                                                            current_agent_propose_enc_out,\
                                                            user_enc_out, user_last_hidden, kb_turn, \
                                                            continue_tm1, act_tm1, slot_tm1, last_hidden,
                                                            cas_continue[t].unsqueeze(0), cas_act[t].unsqueeze(0),
                                                            mode=m)

                continue_tm1 = cas_continue[t].unsqueeze(0)
                act_tm1 = cas_act[t].unsqueeze(0)
                slot_tm1 = agent[2][t].unsqueeze(0)
                agent_continue_proba.append(continue_proba)
                agent_act_proba.append(act_proba)
                agent_slot_proba.append(slot_proba)

            agent_continue_proba = torch.cat(agent_continue_proba, dim=0)  # [T,B,V]
            agent_act_proba = torch.cat(agent_act_proba, dim=0)  # [T,B,V]
            agent_slot_proba = torch.cat(agent_slot_proba, dim=0)  # [T,B,V]
            return (agent_continue_proba, agent_act_proba, agent_slot_proba)

        elif mode == 'test':
            for t in range(self.max_ts):
                continue_proba, act_proba, slot_proba, last_hidden = self.decoder(last_agent_enc_out,
                                                                current_user_request_enc_out, \
                                                                current_user_inform_enc_out, \
                                                                current_agent_request_enc_out, \
                                                                current_agent_propose_enc_out, \
                                                                user_enc_out, user_last_hidden, kb_turn, \
                                                                continue_tm1, act_tm1, slot_tm1, last_hidden,
                                                                mode='test')
                continue_tm1 = continue_proba
                act_tm1 = act_proba
                slot_tm1 = slot_proba
                agent_continue_proba.append(continue_proba)
                agent_act_proba.append(act_proba)
                agent_slot_proba.append(slot_proba)

            agent_continue_proba = torch.cat(agent_continue_proba, dim=0)  # [T,B,V]
            agent_act_proba = torch.cat(agent_act_proba, dim=0)  # [T,B,V]
            agent_slot_proba = torch.cat(agent_slot_proba, dim=0)  # [T,B,V]

            return agent_continue_proba.transpose(1,0).data.cpu().numpy(), \
                   agent_act_proba.transpose(1,0).data.cpu().numpy(), \
                   agent_slot_proba.transpose(1,0).data.cpu().numpy()

    def supervised_loss(self, pred_y, gt_y, continue_mask):
        continue_loss = self.continue_loss( \
            torch.log((pred_y[0]).view(-1, (pred_y[0]).size(2))),
            (gt_y[0]).contiguous().view(-1))
        act_loss = self.act_loss(\
            torch.log((pred_y[1]).view(-1, (pred_y[1]).size(2))),\
            (gt_y[1]).contiguous().view(-1))

        slot_loss = self.slot_loss(pred_y[2], gt_y[2])
        slot_loss = torch.mul(continue_mask, slot_loss)
        slot_loss = torch.mean(slot_loss)
        loss = self.cfg.loss_weights[0]*continue_loss + self.cfg.loss_weights[1]*act_loss + self.cfg.loss_weights[2]*slot_loss
        return loss, continue_loss, act_loss, slot_loss


def get_network(cfg, vocab):
        return Seq2CAS(cfg, vocab)
