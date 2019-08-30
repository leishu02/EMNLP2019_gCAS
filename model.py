import torch
import random
import numpy as np
from config import Config
from reader import Reader
from reader import pad_sequences
from network import get_network
from network import cuda_
from metric import Evaluator

from torch.optim import Adam
from torch.autograd import Variable
import argparse, time, os
import logging


class Model:
    def __init__(self, cfg):
        self.reader = Reader(cfg)
        self.m = get_network(cfg, self.reader.vocab)
        self.EV = Evaluator  # evaluator class
        if cfg.cuda:
            self.m = self.m.cuda()
        self.base_epoch = -1
        self.cfg = cfg


    def _convert_batch(self, py_batch):
        kw_ret = {}
        x = None
        gt_y = None
        if self.cfg.network == 'classification':
            state_vector = py_batch['state_vector']#list of array (batch_size, 1, vector_size)
            batch_size = len(state_vector)
            np_x = np.stack(state_vector, axis=0)#(batch_size, 1, vector_size)
            np_x = np.squeeze(np_x, axis=1)#(batch_size, vector_size)
            np_gt_y = np.zeros((batch_size, self.cfg.output_size), dtype=float)#(batch_size, output_size)
            for batch_id, idx_list in enumerate(py_batch['act_slot_idx_list']):
                for idx in idx_list:
                    np_gt_y[batch_id, idx] = 1.
            x = cuda_(Variable(torch.from_numpy(np_x).float()), self.cfg)
            gt_y = cuda_(Variable(torch.from_numpy(np_gt_y).float()), self.cfg)
        elif 'seq2seq' in self.cfg.network:
            user_np = pad_sequences(py_batch['user_act_seq'], self.cfg.user_max_ts, padding='post', truncating='post').transpose((1, 0))
            last_agent_np = pad_sequences(py_batch['last_agent_act_seq'], self.cfg.agent_max_ts, padding='post', truncating='post').transpose((1, 0))
            current_slot_np = pad_sequences(py_batch['current_slot_seq'], self.cfg.current_slot_max_ts, padding='post', truncating='post').transpose((1, 0))  # (seqlen, batchsize)
            agent_np = pad_sequences(py_batch['agent_act_seq'], self.cfg.agent_max_ts, padding='post', truncating='post').transpose((1, 0))
            user_len = np.array(py_batch['user_len'])
            last_agent_len = np.array(py_batch['last_agent_len'])
            current_slot_len = np.array(py_batch['current_slot_len'])
            agent_len = np.array(py_batch['agent_len'])
            kb_turn_np = np.array(py_batch['kb_turn_vector']).transpose(1, 0, 2)

            kw_ret['user_np'] = user_np #seqlen, batchsize
            kw_ret['last_agent_np'] = last_agent_np#seqlen, batchsize
            kw_ret['current_slot_np'] = current_slot_np#seqlen, batchsize
            kw_ret['agent_np'] = agent_np#seqlen, batchsize
            kw_ret['user_len'] = user_len#batchsize
            kw_ret['last_agent_len'] = last_agent_len#batchsize
            kw_ret['current_slot_len'] = current_slot_len#batchsize
            kw_ret['agent_len'] = agent_len#batchsize
            kw_ret['kb_turn'] = cuda_(Variable(torch.from_numpy(kb_turn_np).float()), self.cfg)#1, batch_size, kb_turn_size
            kw_ret['last_agent'] = cuda_(Variable(torch.from_numpy(last_agent_np).long()), self.cfg)#seqlen, batchsize
            kw_ret['current_slot'] = cuda_(Variable(torch.from_numpy(current_slot_np).long()), self.cfg)#seqlen, batchsize
            x = cuda_(Variable(torch.from_numpy(user_np).long()), self.cfg)#seqlen, batchsize
            gt_y = cuda_(Variable(torch.from_numpy(agent_np).long()), self.cfg)#seqlen, batchsize

        elif 'cas' in self.cfg.network:
            user_np = pad_sequences(py_batch['user_act_seq'], self.cfg.user_max_ts, padding='post', truncating='post').transpose((1, 0))
            last_agent_np = pad_sequences(py_batch['last_agent_act_seq'], self.cfg.agent_max_ts, padding='post', truncating='post').transpose((1, 0))
            current_slot_np = pad_sequences(py_batch['current_slot_seq'], self.cfg.current_slot_max_ts, padding='post', truncating='post').transpose((1, 0))  # (seqlen, batchsize)
            current_user_request_np = pad_sequences(py_batch['current_user_request_seq'], self.cfg.current_singleslot_max_ts, padding='post', truncating='post').transpose((1, 0))  # (seqlen, batchsize)
            current_user_inform_np = pad_sequences(py_batch['current_user_inform_seq'], self.cfg.current_singleslot_max_ts, padding='post', truncating='post').transpose((1, 0))  # (seqlen, batchsize)
            current_agent_request_np = pad_sequences(py_batch['current_agent_request_seq'], self.cfg.current_singleslot_max_ts, padding='post', truncating='post').transpose((1, 0))  # (seqlen, batchsize)
            current_agent_propose_np = pad_sequences(py_batch['current_agent_propose_seq'], self.cfg.current_singleslot_max_ts, padding='post', truncating='post').transpose((1, 0))  # (seqlen, batchsize)
            user_len = np.array(py_batch['user_len'])
            last_agent_len = np.array(py_batch['last_agent_len'])
            current_slot_len = np.array(py_batch['current_slot_len'])
            current_user_request_len = np.array(py_batch['current_user_request_len'])
            current_user_inform_len = np.array(py_batch['current_user_inform_len'])
            current_agent_request_len = np.array(py_batch['current_agent_request_len'])
            current_agent_propose_len = np.array(py_batch['current_agent_propose_len'])
            kb_turn_np = np.array(py_batch['kb_turn_vector']).transpose(1, 0, 2)
            cas_continue_np = np.concatenate(py_batch['cas_continue'], axis=1)#seqlen, batchsize, continuesize
            cas_act_np = np.concatenate(py_batch['cas_act'], axis=1)#seqlen, batchsize, actsize
            cas_slot_np = np.concatenate(py_batch['cas_slot'], axis=1)#seqlen, batchsize, slotsize
            cas_continue_go_np = np.concatenate(py_batch['cas_continue_go'], axis=1)
            cas_act_go_np = np.concatenate(py_batch['cas_act_go'], axis=1)
            cas_slot_go_np = np.concatenate(py_batch['cas_slot_go'], axis=1)
            cas_continue_list_np = np.array(py_batch['cas_continue_list']).transpose((1, 0))  # seqlen, batchsize
            cas_act_list_np = np.array(py_batch['cas_act_list']).transpose((1, 0))#seqlen, batchsize

            kw_ret['user_len'] = user_len#batchsize
            kw_ret['last_agent_len'] = last_agent_len#batchsize
            kw_ret['current_slot_len'] = current_slot_len#batchsize
            kw_ret['current_user_request_len'] = current_user_request_len  # batchsize
            kw_ret['current_user_inform_len'] = current_user_inform_len  # batchsize
            kw_ret['current_agent_request_len'] = current_agent_request_len  # batchsize
            kw_ret['current_agent_propose_len'] = current_agent_propose_len  # batchsize
            kw_ret['kb_turn'] = cuda_(Variable(torch.from_numpy(kb_turn_np).float()), self.cfg)#1, batch_size, kb_turn_size
            kw_ret['last_agent'] = cuda_(Variable(torch.from_numpy(last_agent_np).long()), self.cfg)#seqlen, batchsize
            kw_ret['current_slot'] = cuda_(Variable(torch.from_numpy(current_slot_np).long()), self.cfg)#seqlen, batchsize
            kw_ret['current_user_request'] = cuda_(Variable(torch.from_numpy(current_user_request_np).long()), self.cfg)
            kw_ret['current_user_inform'] = cuda_(Variable(torch.from_numpy(current_user_inform_np).long()), self.cfg)
            kw_ret['current_agent_request'] = cuda_(Variable(torch.from_numpy(current_agent_request_np).long()), self.cfg)
            kw_ret['current_agent_propose'] = cuda_(Variable(torch.from_numpy(current_agent_propose_np).long()), self.cfg)
            kw_ret['continue_go'] = cuda_(Variable(torch.from_numpy(cas_continue_go_np).float()), self.cfg)
            kw_ret['act_go'] = cuda_(Variable(torch.from_numpy(cas_act_go_np).float()), self.cfg)
            kw_ret['slot_go'] = cuda_(Variable(torch.from_numpy(cas_slot_go_np).float()), self.cfg)
            kw_ret['cas_act'] = cuda_(Variable(torch.from_numpy(cas_act_np).float()), self.cfg)
            kw_ret['cas_continue'] = cuda_(Variable(torch.from_numpy(cas_continue_np).float()), self.cfg)

            cas_continue_list = cuda_(Variable(torch.from_numpy(cas_continue_list_np).long()), self.cfg)
            cas_act_list = cuda_(Variable(torch.from_numpy(cas_act_list_np).long()), self.cfg)
            cas_slot = cuda_(Variable(torch.from_numpy(cas_slot_np).float()), self.cfg)

            x = cuda_(Variable(torch.from_numpy(user_np).long()), self.cfg)#seqlen, batchsize
            gt_y = (cas_continue_list, cas_act_list, cas_slot)
        else:
            assert()
        return x, gt_y, kw_ret

    def train(self):
        lr = self.cfg.lr
        prev_min_loss = np.inf
        prev_max_metrics = 0.
        early_stop_count = self.cfg.early_stop_count
        train_time = 0
        for epoch in range(self.cfg.epoch_num):
            sw = time.time()
            if epoch <= self.base_epoch:
                continue
            sup_loss = 0
            sup_cnt = 0
            data_iterator = self.reader.mini_batch_iterator('train')
            optim = Adam(lr=lr, params=filter(lambda x: x.requires_grad, self.m.parameters()), weight_decay=1e-5)
            for iter_num, dial_batch in enumerate(data_iterator):
                for turn_num, turn_batch in enumerate(dial_batch):
                    if self.cfg.truncated:
                        logging.debug('iter %d turn %d' % (iter_num, turn_num))
                    optim.zero_grad()
                    x, gt_y, kw_ret = self._convert_batch(turn_batch)
                    if 'cas' in self.cfg.network:
                        loss, continue_loss, act_loss, slot_loss = self.m(x=x, gt_y=gt_y, mode='train', **kw_ret)
                    else:
                        loss = self.m(x=x, gt_y=gt_y, mode='train', **kw_ret)
                    loss.backward(retain_graph=turn_num != len(dial_batch) - 1)
                    grad = torch.nn.utils.clip_grad_norm_(self.m.parameters(), self.cfg.grad_clip_norm)
                    optim.step()
                    sup_loss += loss.item()
                    sup_cnt += 1
                    if 'cas' in self.cfg.network:
                        logging.debug('loss:{} continue_loss:{} act_loss:{} slot_loss:{} grad:{}'.format(loss.item(),\
                                                                                                continue_loss.item(),\
                                                                                                act_loss.item(),\
                                                                                                slot_loss.item(),\
                                                                                                grad))
                    else:
                        logging.debug('loss:{} grad:{}'.format(loss.item(), grad))

            epoch_sup_loss = sup_loss / (sup_cnt + 1e-8)
            train_time += time.time() - sw
            logging.info('Traning time: {}'.format(train_time))
            logging.info('avg training loss in epoch %d sup:%f' % (epoch, epoch_sup_loss))

            valid_sup_loss, valid_unsup_loss = self.validate()
            logging.info('validation loss in epoch %d sup:%f unsup:%f' % (epoch, valid_sup_loss, valid_unsup_loss))
            logging.info('time for epoch %d: %f' % (epoch, time.time() - sw))
            valid_loss = valid_sup_loss + valid_unsup_loss

            metrics = self.eval(data='dev')
            valid_metrics = metrics[-1]
            logging.info('valid metric %f ' %(valid_metrics))
            #if valid_loss <= prev_min_loss:
            if valid_metrics >= prev_max_metrics:
                self.save_model(epoch)
                #prev_min_loss = valid_loss
                prev_max_metrics = valid_metrics
                early_stop_count = self.cfg.early_stop_count
            else:
                early_stop_count -= 1
                lr *= self.cfg.lr_decay
                if not early_stop_count:
                    break
                logging.info('early stop countdown %d, learning rate %f' % (early_stop_count, lr))

    def run_metrics(self, data='test'):
        if os.path.exists(self.cfg.result_path):
            self.m.eval()
            ev = self.EV(self.cfg)
            res = ev.run_metrics()
            self.m.train()
        else:
            self.eval(data='test')
        return res

    def eval(self, data='test'):
        self.m.eval()
        self.reader.result_file = None
        data_iterator = self.reader.mini_batch_iterator(data)
        mode = 'test'
        for batch_num, dial_batch in enumerate(data_iterator):
            for turn_num, turn_batch in enumerate(dial_batch):
                x, gt_y, kw_ret = self._convert_batch(turn_batch)
                pred_y = self.m(x=x, gt_y=gt_y, mode=mode, **kw_ret)
                self.reader.wrap_result(turn_batch, pred_y)
        if self.reader.result_file != None:
            self.reader.result_file.close()
        ev = self.EV(self.cfg)
        res = ev.run_metrics()
        self.m.train()
        return res

    def validate(self, data='dev'):
        self.m.eval()
        data_iterator = self.reader.mini_batch_iterator(data)
        sup_loss, unsup_loss = 0, 0
        sup_cnt, unsup_cnt = 0, 0
        for dial_batch in data_iterator:
            for turn_num, turn_batch in enumerate(dial_batch):
                x, gt_y, kw_ret = self._convert_batch(turn_batch)
                if 'cas' in self.cfg.network:
                    loss, continue_loss, act_loss, slot_loss = self.m(x=x, gt_y=gt_y, mode='train', **kw_ret)
                else:
                    loss = self.m(x=x, gt_y=gt_y, mode='train', **kw_ret)
                sup_loss += loss.item()
                sup_cnt += 1
                if 'cas' in self.cfg.network:
                    logging.debug('loss:{} continue_loss:{} act_loss:{} slot_loss:{}'.format(loss.item(),
                                                                                                  continue_loss.item(),
                                                                                                  act_loss.item(),
                                                                                                  slot_loss.item()))
                else:
                    logging.debug('loss:{}'.format(loss.item()))

        sup_loss /= (sup_cnt + 1e-8)
        unsup_loss /= (unsup_cnt + 1e-8)
        self.m.train()
        return sup_loss, unsup_loss

    def save_model(self, epoch, path=None):
        if not path:
            path = self.cfg.model_path
        all_state = {'lstd': self.m.state_dict(),
                     'config': self.cfg.__dict__,
                     'epoch': epoch}
        torch.save(all_state, path)

    def load_model(self, path=None):
        if not path:
            path = self.cfg.model_path
        all_state = torch.load(path)
        self.m.load_state_dict(all_state['lstd'])
        self.base_epoch = all_state.get('epoch', 0)


    def freeze_module(self, module):
        for param in module.parameters():
            param.requires_grad = False

    def unfreeze_module(self, module):
        for param in module.parameters():
            param.requires_grad = True

    def count_params(self):
        module_parameters = filter(lambda p: p.requires_grad, self.m.parameters())
        param_cnt = sum([np.prod(p.size()) for p in module_parameters if p.requires_grad == True])
        print('total trainable params: %d' % param_cnt)
        print(self.m)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-domain')
    parser.add_argument('-network')
    parser.add_argument('-mode')
    parser.add_argument('-cfg', nargs='*')
    args = parser.parse_args()

    cfg = Config(args.domain)
    cfg.init_handler(args.network)
    if args.cfg:
        for pair in args.cfg:
            k, v = tuple(pair.split('='))
            dtype = type(getattr(cfg, k))
            if dtype == type(None):
                raise ValueError()
            if dtype is bool:
                v = False if v == 'False' else True
            else:
                v = dtype(v)
            setattr(cfg, k, v)

    logging.debug(str(cfg))
    if cfg.cuda:
        torch.cuda.set_device(cfg.cuda_device)
        logging.debug('Device: {}'.format(torch.cuda.current_device()))
    cfg.mode = args.mode

    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    m = Model(cfg)
    m.count_params()
    if args.mode == 'train':#train the model from scratch
        m.train()
        m.load_model()
        m.eval(data='test')
    elif args.mode == 'adjust':#continue to train the model
        m.load_model()
        m.train()
    elif args.mode == 'test':#test the model, save the result
        m.load_model()
        m.eval(data='test')
    elif args.mode == 'eval':#evaluation the testing result
        m.load_model()
        m.run_metrics(data='test')


if __name__ == '__main__':
    main()
