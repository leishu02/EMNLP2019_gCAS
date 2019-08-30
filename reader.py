import numpy as np
import cPickle as pickle
from nltk.tokenize import word_tokenize
import logging
import random
import os
import csv
import json


def text_to_dict(path):
    """ Read in a text file as a dictionary where keys are text and values are indices (line numbers) """
    slot_set = {}
    with open(path, 'r') as f:
        index = 0
        for line in f.readlines():
            slot_set[line.strip('\n').strip('\r')] = index
            index += 1
    return slot_set


class _ReaderBase(object):
    class LabelSet:
        def __init__(self):
            self._idx2item = {}
            self._item2idx = {}
            self._freq_dict = {}

        def __len__(self):
            return len(self._idx2item)

        def _absolute_add_item(self, item):
            idx = len(self)
            self._idx2item[idx] = item
            self._item2idx[item] = idx

        def add_item(self, item):
            if item not in self._freq_dict:
                self._freq_dict[item] = 0
            self._freq_dict[item] += 1

        def construct(self, limit=None):
            l = sorted(self._freq_dict.keys(), key=lambda x: -self._freq_dict[x])
            print('Actual label size %d' % (len(l) + len(self._idx2item)))
            if limit == None:
                limit = len(l) + len(self._idx2item)
            if len(l) + len(self._idx2item) < limit:
                logging.warning('actual label set smaller than that configured: {}/{}'
                                .format(len(l) + len(self._idx2item), limit))
            for item in l:
                if item not in self._item2idx:
                    idx = len(self._idx2item)
                    self._idx2item[idx] = item
                    self._item2idx[item] = idx
                    if len(self._idx2item) >= limit:
                        break

        def encode(self, item):
            return self._item2idx[item]

        def decode(self, idx):
            return self._idx2item[idx]

    class Vocab(LabelSet):
        def __init__(self, init=True):
            _ReaderBase.LabelSet.__init__(self)
            if init:
                self._absolute_add_item('<pad>')  # 0
                self._absolute_add_item('<go>')  # 1
                self._absolute_add_item('<unk>')  # 2
                self._absolute_add_item('EOS_U')  # 3 eos user
                self._absolute_add_item('EOS_A')  # 4 eos last agent
                self._absolute_add_item('EOS')  # 5
                self._absolute_add_item('EOS_C')  # 6 eos current slots

        def load_vocab(self, vocab_path):
            f = open(vocab_path, 'rb')
            dic = pickle.load(f)
            self._idx2item = dic['idx2item']
            self._item2idx = dic['item2idx']
            self._freq_dict = dic['freq_dict']
            f.close()

        def save_vocab(self, vocab_path):
            f = open(vocab_path, 'wb')
            dic = {
                'idx2item': self._idx2item,
                'item2idx': self._item2idx,
                'freq_dict': self._freq_dict
            }
            pickle.dump(dic, f)
            f.close()

        def sentence_encode(self, word_list):
            return [self.encode(_) for _ in word_list]

        def sentence_decode(self, index_list, eos=None):
            l = [self.decode(_) for _ in index_list]
            if not eos or eos not in l:
                return ' '.join(l)
            else:
                idx = l.index(eos)
                return ' '.join(l[:idx])

        def nl_decode(self, l, eos=None):
            return [self.sentence_decode(_, eos) + '\n' for _ in l]

        def encode(self, item):
            if item in self._item2idx:
                return self._item2idx[item]
            else:
                return self._item2idx['<unk>']

        def decode(self, idx):
            if idx < len(self):
                return self._idx2item[idx]
            else:
                if self.cfg.vocab_size != None:
                    return 'ITEM_%d' % (idx - self.cfg.vocab_size)

    def __init__(self, cfg):
        self.train, self.dev, self.test = [], [], []
        self.vocab = self.Vocab()
        self.result_file = ''
        self.cfg = cfg

    def _construct(self, *args):
        """
        load data, construct vocab and store them in self.train/dev/test
        :param args:
        :return:
        """
        raise NotImplementedError('This is an abstract class, bro')

    def _bucket_by_turn(self, encoded_data):
        turn_bucket = {}
        for dial in encoded_data:
            turn_len = len(dial)
            if turn_len not in turn_bucket:
                turn_bucket[turn_len] = []
            turn_bucket[turn_len].append(dial)
        del_l = []
        for k in turn_bucket:
            if k >= 5: del_l.append(k)
            logging.debug("bucket %d instance %d" % (k, len(turn_bucket[k])))
        # for k in del_l:
        #    turn_bucket.pop(k)
        return turn_bucket

    def _mark_batch_as_supervised(self, all_batches):
        supervised_num = int(len(all_batches) * self.cfg.spv_proportion / 100)
        for i, batch in enumerate(all_batches):
            for dial in batch:
                for turn in dial:
                    turn['supervised'] = i < supervised_num
                    if not turn['supervised']:
                        turn['degree'] = [0.] * self.cfg.degree_size  # unsupervised learning. DB degree should be unknown
        return all_batches

    def _construct_mini_batch(self, data):
        all_batches = []
        batch = []
        for dial in data:
            batch.append(dial)
            if len(batch) == self.cfg.batch_size:
                all_batches.append(batch)
                batch = []

        if len(batch) > 0.5 * self.cfg.batch_size:
            all_batches.append(batch)
        elif len(all_batches):
            all_batches[-1].extend(batch)
        else:
            all_batches.append(batch)
        return all_batches

    def _transpose_batch(self, batch):
        dial_batch = []
        turn_num = len(batch[0])
        for turn in range(turn_num):
            turn_l = {}
            for dial in batch:
                this_turn = dial[turn]
                for k in this_turn:
                    if k not in turn_l:
                        turn_l[k] = []
                    turn_l[k].append(this_turn[k])
            dial_batch.append(turn_l)
        return dial_batch

    def mini_batch_iterator(self, set_name):
        name_to_set = {'train': self.train, 'test': self.test, 'dev': self.dev}
        dial = name_to_set[set_name]
        turn_bucket = self._bucket_by_turn(dial)
        all_batches = []
        for k in turn_bucket:
            batches = self._construct_mini_batch(turn_bucket[k])
            all_batches += batches
        self._mark_batch_as_supervised(all_batches)
        random.shuffle(all_batches)
        for i, batch in enumerate(all_batches):
            yield self._transpose_batch(batch)

    def wrap_result(self, turn_batch, pred_y):
        raise NotImplementedError('This is an abstract class, bro')


class Reader(_ReaderBase):
    def __init__(self, cfg):
        super(Reader, self).__init__(cfg)
        self._construct()
        self.result_file = ''

    def _get_tokenized_data(self, raw_data, construct_vocab, remove_slot_value):
        tokenized_data = []
        for dial_id, dial in raw_data.items():
            tokenized_dial = []
            last_agent_act = []
            last_agent_act_seq = []
            for turn_num, turn in enumerate(dial):
                state = turn['state']
                agent_act = turn['agent_act']
                user_act = turn['user_act']
                user_act_seq = self.prepare_sequence_from_act(user_act, remove_slot_value)
                agent_act_seq = self.prepare_sequence_from_act(agent_act, remove_slot_value)
                current_slot_seqs, current_slot_seq, kb_turn_vector = self.prepare_state_sequence(state, remove_slot_value)
                act_slot_pairs, act_slot_idx_list = self.prepare_act_slot_pairs(agent_act)
                tokenized_dial.append({
                    'dial_id': dial_id,
                    'turn_num': turn_num,
                    'state': state,
                    'user_act': user_act,
                    'user_act_seq': user_act_seq+['EOS_U'],
                    'last_agent_act': last_agent_act,
                    'last_agent_act_seq': last_agent_act_seq+['EOS_A'],
                    'agent_act': agent_act,
                    'agent_act_seq': agent_act_seq+['EOS'],
                    'kb_turn_vector': kb_turn_vector,
                    'current_slot_seq': current_slot_seq+['EOS_C'],
                    'act_slot_pairs': act_slot_pairs,
                    'current_slot_seqs': [c+['EOS_C'] for c in current_slot_seqs],
                })
                last_agent_act = agent_act
                last_agent_act_seq = agent_act_seq
                if construct_vocab:
                    for word in user_act_seq+current_slot_seq+last_agent_act_seq+agent_act_seq:
                        self.vocab.add_item(word)
            tokenized_data.append(tokenized_dial)
        return tokenized_data

    def prepare_cas(self, agent_act):
        continues_np = np.zeros((self.cfg.cas_max_ts, 1, self.cfg.continue_size), dtype=float)
        acts_np = np.zeros((self.cfg.cas_max_ts, 1, self.cfg.act_size), dtype=float)
        slots_np = np.zeros((self.cfg.cas_max_ts, 1, self.cfg.slot_size), dtype=float)
        act_list = [0]*self.cfg.cas_max_ts
        continue_list = [0]*self.cfg.cas_max_ts
        for i, (act, slots) in enumerate(agent_act):
            act = act.lower()
            continues_np[i][0][self.continue2idx['<continue>']] = 1.0
            continue_list[i] = self.continue2idx['<continue>']
            if len(slots) == 0:
                acts_np[i][0][self.act2idx[act]] = 1.0
                act_list[i] = self.act2idx[act]
            else:
                if act == 'request':
                    for k, v in slots.items():
                        k = k.lower()
                        v = v.lower()
                        if k in self.slot_set:
                            if k != 'taskcomplete' and v == '':
                                acts_np[i][0][self.act2idx[act]] = 1.0
                                act_list[i] = self.act2idx[act]
                                slots_np[i][0][self.slot2idx[k]] = 1.0
                            else:
                                acts_np[i][0][self.act2idx[act]] = 1.0
                                act_list[i] = self.act2idx[act]
                                if k+'=value' not in self.slot2idx:
                                    self.slot2idx[k+'=value'] = len(self.slot2idx)
                                slots_np[i][0][self.slot2idx[k+'=value']] = 1.0
                        else:
                            pass
                elif act == 'multiple_choice':
                    for k, v in slots.items():
                        k = k.lower()
                        v = v.lower()
                        if k in self.slot_set:
                            if v == '' or k == 'mc_list':
                                acts_np[i][0][self.act2idx[act]] = 1.0
                                act_list[i] = self.act2idx[act]
                                slots_np[i][0][self.slot2idx[k]] = 1.0
                            else:
                                acts_np[i][0][self.act2idx[act]] = 1.0
                                act_list[i] = self.act2idx[act]
                                if k+'=value' not in self.slot2idx:
                                    self.slot2idx[k+'=value'] = len(self.slot2idx)
                                slots_np[i][0][self.slot2idx[k + '=value']] = 1.0
                elif act == 'inform':
                    for k, v in slots.items():
                        k = k.lower()
                        if k in self.slot_set:
                            acts_np[i][0][self.act2idx[act]] = 1.0
                            act_list[i] = self.act2idx[act]
                            if k + '=value' not in self.slot2idx:
                                self.slot2idx[k + '=value'] = len(self.slot2idx)
                            slots_np[i][0][self.slot2idx[k + '=value']] = 1.0
                else:
                    for k, v in slots.items():
                        k = k.lower()
                        if k in self.slot_set:
                            acts_np[i][0][self.act2idx[act]] = 1.0
                            act_list[i] = self.act2idx[act]
                            slots_np[i][0][self.slot2idx[k]] = 1.0
        #pad
        if len(agent_act) < self.cfg.cas_max_ts:
            continues_np[len(agent_act)][0][self.continue2idx['<stop>']] = 1.0
            continue_list[len(agent_act)] = self.continue2idx['<stop>']
        for i in range(len(agent_act), self.cfg.cas_max_ts):
            #continue is all zero
            if i > len(agent_act):
                continues_np[i][0][self.continue2idx['<pad>']] = 1.0
                continue_list[i] = self.continue2idx['<pad>']
            acts_np[i][0][self.act2idx['<pad>']] = 1.0
            #slots_np[i][0][self.act2idx['<pad>']] = 1.0

        return continues_np, acts_np, slots_np, continue_list, act_list

    def _get_cas_encoded_data(self, tokenized_data):
        encoded_data = []
        max_ts = 0
        continue_go = np.zeros((1, 1, self.cfg.continue_size), dtype=float)
        act_go = np.zeros((1, 1, self.cfg.act_size), dtype=float)
        slot_go = np.zeros((1, 1, self.cfg.slot_size), dtype=float)
        continue_go[0][0][self.continue2idx['<go>']] = 1.0
        act_go[0][0][self.act2idx['<go>']] = 1.0
        slot_go[0][0][self.slot2idx['<go>']] = 1.0
        for dial in tokenized_data:
            encoded_dial = []
            for turn in dial:
                continue_np, act_np, slot_np, continue_list, act_list = self.prepare_cas(turn['agent_act'])
                current_slot_seqs = turn['current_slot_seqs']
                current_user_request_seq = current_slot_seqs[0]
                current_user_inform_seq = current_slot_seqs[2]
                current_agent_request_seq = current_slot_seqs[1]
                current_agent_propose_seq = current_slot_seqs[3]
                encoded_dial.append({
                    'dial_id': turn['dial_id'],
                    'turn_num': turn['turn_num'],
                    'state': turn['state'],
                    'user_act': turn['user_act'],
                    'user_act_seq': self.vocab.sentence_encode(turn['user_act_seq']),
                    'user_len': len(turn['user_act_seq']),
                    'last_agent_act': turn['last_agent_act'],
                    'last_agent_act_seq': self.vocab.sentence_encode(turn['last_agent_act_seq']),
                    'last_agent_len': len(turn['last_agent_act_seq']),
                    'agent_act': turn['agent_act'],
                    'agent_act_seq': self.vocab.sentence_encode(turn['agent_act_seq']),
                    'agent_len': len(turn['agent_act_seq']),
                    'act_slot_pairs': turn['act_slot_pairs'],
                    'kb_turn_vector': turn['kb_turn_vector'],
                    'current_slot_seq': self.vocab.sentence_encode(turn['current_slot_seq']),
                    'current_slot_len': len(turn['current_slot_seq']),
                    'cas_continue': continue_np,
                    'cas_act': act_np,
                    'cas_slot': slot_np,
                    'cas_continue_go': continue_go,
                    'cas_act_go': act_go,
                    'cas_slot_go': slot_go,
                    'cas_act_list': act_list,
                    'cas_continue_list': continue_list,
                    'current_user_request_seq': self.vocab.sentence_encode(current_user_request_seq),
                    'current_user_request_len': len(current_user_request_seq),
                    'current_user_inform_seq': self.vocab.sentence_encode(current_user_inform_seq),
                    'current_user_inform_len': len(current_user_inform_seq),
                    'current_agent_request_seq': self.vocab.sentence_encode(current_agent_request_seq),
                    'current_agent_request_len': len(current_agent_request_seq),
                    'current_agent_propose_seq':self.vocab.sentence_encode(current_agent_propose_seq),
                    'current_agent_propose_len': len(current_agent_propose_seq),
                })

                if max(len(current_user_request_seq), len(current_user_inform_seq),
                       len(current_agent_request_seq), len(current_agent_propose_seq)) > max_ts:
                    max_ts = max(len(current_user_request_seq), len(current_user_inform_seq),
                       len(current_agent_request_seq), len(current_agent_propose_seq))
            encoded_data.append(encoded_dial)
        print (max_ts)
        return encoded_data

    def _split_data(self, encoded_data, split):
        """
        split data into train/dev/test
        :param encoded_data: list
        :param split: tuple / list
        :return:
        """
        total = sum(split)
        assert (total == len(encoded_data))
        train, dev, test = encoded_data[:split[0]], encoded_data[split[0]:split[0] + split[1]], encoded_data[
                                                                                                split[0] + split[1]:]
        return train, dev, test

    def prepare_sequence_from_act(self, act, remove_slot_value):
        act_seq = []
        for act, slot in act:
            act_seq.append(act)
            act_seq.append('(')
            for k, v in slot.items():
                act_seq.append(k.lower())
                if v != '':
                    act_seq.append('=')
                    if not remove_slot_value:
                        act_seq += word_tokenize(v.lower())
                act_seq.append(';')
            if act_seq[-1] == ';':
                act_seq = act_seq[:-1]
            act_seq.append(')')
        return act_seq

    def prepare_state_sequence(self, state, remove_slot_value):
        """ Create the representation for each state """
        kb_results_dict = state['kb_results_dict']
        current_slot_seqs = []
        for k_c, v_c in state['current_slots'].items():
            current_slot_seq = []
            current_slot_seq += word_tokenize(k_c.lower())
            current_slot_seq += [':', '{']
            for k, v in v_c.items():
                current_slot_seq += word_tokenize(k.lower())
                if not remove_slot_value:
                    current_slot_seq += [':']
                    current_slot_seq += word_tokenize(v.lower())
                current_slot_seq += [',']
            if current_slot_seq[-1] == ',':
                current_slot_seq = current_slot_seq[:-1]
            current_slot_seq += ['}']
            current_slot_seqs.append(current_slot_seq)
        current_slot_seq = []
        for c in current_slot_seqs:
            current_slot_seq += c



        ########################################################################
        #  One-hot representation of the turn count?
        ########################################################################
        turn_rep = np.zeros((1, 1)) + state['turn'] / 10.
        turn_onehot_rep = np.zeros((1, self.max_turn))
        turn_onehot_rep[0, state['turn']] = 1.0

        ########################################################################
        #   Representation of KB results (scaled counts)
        ########################################################################
        kb_count_rep = np.zeros((1, self.slot_cardinality + 1)) + kb_results_dict['matching_all_constraints'] / 100.
        for slot in kb_results_dict:
            if slot in self.slot_set:
                kb_count_rep[0, self.slot_set[slot]] = kb_results_dict[slot] / 100.

        ########################################################################
        #   Representation of KB results (binary)
        ########################################################################
        kb_binary_rep = np.zeros((1, self.slot_cardinality + 1)) + np.sum(
            kb_results_dict['matching_all_constraints'] > 0.)
        for slot in kb_results_dict:
            if slot in self.slot_set:
                kb_binary_rep[0, self.slot_set[slot]] = np.sum(kb_results_dict[slot] > 0.)

        turn_kb_representation = np.hstack([turn_rep, turn_onehot_rep, kb_binary_rep, kb_count_rep])
        return current_slot_seqs, current_slot_seq, turn_kb_representation

    def prepare_state_representation(self, state):
        """ Create the representation for each state """
        user_action = state['user_action']
        current_slots = state['current_slots']
        kb_results_dict = state['kb_results_dict']
        agent_last = state['agent_action']

        ########################################################################
        #   Create one-hot of acts to represent the current user action
        ########################################################################
        user_act_rep = np.zeros((1, self.act_cardinality))
        if user_action['diaact'].lower() in self.act_set:
            user_act_rep[0, self.act_set[user_action['diaact'].lower()]] = 1.0

        ########################################################################
        #     Create bag of inform slots representation to represent the current user action
        ########################################################################
        user_inform_slots_rep = np.zeros((1, self.slot_cardinality))
        for slot in user_action['inform_slots'].keys():
            if slot.lower() in self.slot_set:
                user_inform_slots_rep[0, self.slot_set[slot.lower()]] = 1.0

        ########################################################################
        #   Create bag of request slots representation to represent the current user action
        ########################################################################
        user_request_slots_rep = np.zeros((1, self.slot_cardinality))
        for slot in user_action['request_slots'].keys():
            if slot.lower() in self.slot_set:
                user_request_slots_rep[0, self.slot_set[slot.lower()]] = 1.0

        ########################################################################
        #   Creat bag of filled_in slots based on the current_slots
        ########################################################################
        current_slots_rep = np.zeros((1, self.slot_cardinality))
        for slot in current_slots['inform_slots']:
            if slot.lower() in self.slot_set:
                current_slots_rep[0, self.slot_set[slot.lower()]] = 1.0

        ########################################################################
        #   Encode last agent act
        ########################################################################
        agent_act_rep = np.zeros((1, self.act_cardinality))
        if agent_last:
            if agent_last['diaact'].lower() in self.act_set:
                agent_act_rep[0, self.act_set[agent_last['diaact'].lower()]] = 1.0

        ########################################################################
        #   Encode last agent inform slots
        ########################################################################
        agent_inform_slots_rep = np.zeros((1, self.slot_cardinality))
        if agent_last:
            for slot in agent_last['inform_slots'].keys():
                if slot.lower() in self.slot_set:
                    agent_inform_slots_rep[0, self.slot_set[slot]] = 1.0

        ########################################################################
        #   Encode last agent request slots
        ########################################################################
        agent_request_slots_rep = np.zeros((1, self.slot_cardinality))
        if agent_last:
            for slot in agent_last['request_slots'].keys():
                if slot.lower() in self.slot_set:
                    agent_request_slots_rep[0, self.slot_set[slot]] = 1.0

        turn_rep = np.zeros((1, 1)) + state['turn'] / 10.

        ########################################################################
        #  One-hot representation of the turn count?
        ########################################################################
        turn_onehot_rep = np.zeros((1, self.max_turn))
        turn_onehot_rep[0, state['turn']] = 1.0

        ########################################################################
        #   Representation of KB results (scaled counts)
        ########################################################################
        kb_count_rep = np.zeros((1, self.slot_cardinality + 1)) + kb_results_dict['matching_all_constraints'] / 100.
        for slot in kb_results_dict:
            if slot in self.slot_set:
                kb_count_rep[0, self.slot_set[slot]] = kb_results_dict[slot] / 100.

        ########################################################################
        #   Representation of KB results (binary)
        ########################################################################
        kb_binary_rep = np.zeros((1, self.slot_cardinality + 1)) + np.sum(
            kb_results_dict['matching_all_constraints'] > 0.)
        for slot in kb_results_dict:
            if slot in self.slot_set:
                kb_binary_rep[0, self.slot_set[slot]] = np.sum(kb_results_dict[slot] > 0.)

        final_representation = np.hstack(
            [user_act_rep, user_inform_slots_rep, user_request_slots_rep, agent_act_rep, agent_inform_slots_rep,
             agent_request_slots_rep, current_slots_rep, turn_rep, turn_onehot_rep, kb_binary_rep, kb_count_rep])
        return final_representation

    def prepare_act_slot_pairs(self, agent_act):
        def add_act_slot_pair(action, slot):
            l = len(self.act_slot_pair_dict)  # type: int
            pair = action.lower() + '+' + slot.lower()
            if pair not in self.act_slot_pair_dict:
                self.act_slot_pair_dict[pair] = l
            return pair

        output_act_slot_pairs = []
        for (act, slots) in agent_act:
            if len(slots) == 0:
                output_act_slot_pairs.append(add_act_slot_pair(act, ''))
            else:
                if act == 'request':
                    for k, v in slots.items():
                        if k != 'taskcomplete' and v == '':
                            if k.lower() in self.slot_set:
                                output_act_slot_pairs.append(add_act_slot_pair(act, k.lower()))
                        else:
                            if k.lower() in self.slot_set:
                                output_act_slot_pairs.append(add_act_slot_pair('inform', k.lower()))
                elif act == 'multiple_choice':
                    for k, v in slots.items():
                        if v == '' or k == 'mc_list':
                            if k.lower() in self.slot_set:
                                output_act_slot_pairs.append(add_act_slot_pair(act, k.lower()))
                        else:
                            if k.lower() in self.slot_set:
                                output_act_slot_pairs.append(add_act_slot_pair('inform', k.lower()))
                elif act == 'inform':
                    for k, v in slots.items():
                        if k.lower() in self.slot_set:
                            output_act_slot_pairs.append(add_act_slot_pair(act, k.lower()))
                else:
                    for k, v in slots.items():
                        if k.lower() in self.slot_set:
                            output_act_slot_pairs.append(add_act_slot_pair(act, k.lower()))
        output_act_slot_idx_list = [self.act_slot_pair_dict[o] for o in output_act_slot_pairs]
        return output_act_slot_pairs, list(set(output_act_slot_idx_list))

    def _construct(self):
        """
        construct encoded train, dev, test set.
        :param data_json_path:
        :param db_json_path:
        :return:
        """
        raw_data = pickle.load(open(self.cfg.dialog_path, 'rb'))
        self.act_set = text_to_dict(self.cfg.act_path)
        self.slot_set = text_to_dict(self.cfg.slot_path)
        self.act_cardinality = len(self.act_set.keys())
        self.slot_cardinality = len(self.slot_set.keys())
        self.act_slot_pair_dict = dict()
        self.max_turn = self.cfg.max_turn

        construct_vocab = True
        if not os.path.isfile(self.cfg.vocab_path):
            construct_vocab = True
            print('Constructing vocab file...')
        tokenized_data = self._get_tokenized_data(raw_data, construct_vocab, self.cfg.remove_slot_value)
        if construct_vocab:
            self.vocab.construct(self.cfg.vocab_size)
            self.vocab.save_vocab(self.cfg.vocab_path)
        else:
            self.vocab.load_vocab(self.cfg.vocab_path)

        if 'cas' in self.cfg.network:
            self.continue2idx = {'<pad>': 0, '<go>': 1, '<continue>': 2, '<stop>':3}
            self.act2idx = {'<pad>': 0, '<go>': 1}
            for a in self.act_set:
                self.act2idx[a] = len(self.act2idx)
            self.slot2idx = {'<go>': 0}
            for s in self.slot_set:
                self.slot2idx[s] = len(self.slot2idx)
            encoded_data = self._get_cas_encoded_data(tokenized_data)
            self.idx2continue = {v:k for k, v in self.continue2idx.items()}
            self.idx2act = {v:k for k, v in self.act2idx.items()}
            self.idx2slot = {v:k for k, v in self.slot2idx.items()}
            assert(len(self.continue2idx) == self.cfg.continue_size and len(self.slot2idx) == self.cfg.slot_size
                   and len(self.act2idx) == self.cfg.act_size)

        self.train, self.dev, self.test = self._split_data(encoded_data, self.cfg.split)
        random.shuffle(self.train)
        random.shuffle(self.dev)
        random.shuffle(self.test)

    def wrap_result(self, turn_batch, pred_y):
        def _map_cas_to_pair(p_continue, p_act, p_slot):
            pred_act_slot_pair = []
            pred_act_slots = []
            for i in range(p_continue.shape[0]):  # seqlen
                c = self.idx2continue[np.argmax(p_continue[i])]
                if c == '<continue>':  # continue
                    a_idx = np.argmax(p_act[i])
                    act = self.idx2act[a_idx]
                    if act in self.act_set:
                        s = p_slot[i]
                        slot_idx = np.argwhere(s >= 0.5).flatten()
                        cand_slots = [self.idx2slot[si] for si in slot_idx if si != 0]# slot != <go>
                        slots = {}
                        if len(cand_slots) == 0:
                            pred_act_slot_pair.append(act+'+')
                        else:
                            if act == 'request':
                                for cand_s in cand_slots:
                                    if '=' in cand_s:#with value
                                        k, v = cand_s.split('=')
                                    else:
                                        k = cand_s
                                        v = ''
                                    if k in self.slot_set:
                                        if k != 'taskcomplete' and v == '':
                                            slots[k] = ''
                                            pred_act_slot_pair.append(act + '+'+k)
                                        else:
                                            slots[k] = 'SOMEVALUE'
                                            pred_act_slot_pair.append('inform' + '+' + k)
                                    else:#k does not exist
                                        pass
                            elif act == 'multiple_choice':
                                for cand_s in cand_slots:
                                    if '=' in cand_s:#with value
                                        k, v = cand_s.split('=')
                                    else:
                                        k = cand_s
                                        v = ''
                                    if k in self.slot_set:
                                        if v == '' or k == 'mc_list':
                                            slots[k] = ''
                                            pred_act_slot_pair.append(act + '+'+k)
                                        else:
                                            slots[k] = 'SOMEVALUE'
                                            pred_act_slot_pair.append('inform' + '+' + k)
                                    else:
                                        pass
                            elif act == 'inform':
                                for cand_s in cand_slots:
                                    if '=' in cand_s:#with value
                                        k, v = cand_s.split('=')
                                    else:
                                        k = cand_s
                                        v = ''
                                    if k in self.slot_set:
                                        slots[k] = 'SOMEVALUE'
                                        pred_act_slot_pair.append('inform' + '+' + k)
                            else:
                                for cand_s in cand_slots:
                                    if '=' in cand_s:#with value
                                        k, v = cand_s.split('=')
                                    else:
                                        k = cand_s
                                        v = ''
                                    if k in self.slot_set:
                                        slots[k] = ''
                                        pred_act_slot_pair.append(act + '+' + k)
                        pred_act_slots.append((act, slots))
                    else:  # act is not in act set
                        pass
                else:  # stop
                    break
            return list(pred_act_slot_pair), pred_act_slots

        field = ['dial_id', 'turn_num', 'agent_act', 'act_slot_pairs', 'pred_act_slot_pairs', 'pred_agent_act',
                 'pred_agent_act_seq', 'state']
        results = []
        batch_size = len(turn_batch['state'])
        for i in range(batch_size):
            entry = {}
            if 'cas' in self.cfg.network:
                act_slot_pairs, act_slot_list = _map_cas_to_pair(pred_y[0][i], pred_y[1][i], pred_y[2][i])
                entry['pred_act_slot_pairs'] = json.dumps(act_slot_pairs)
                entry['pred_agent_act'] = json.dumps(act_slot_list)
                entry['pred_agent_act_seq'] = json.dumps([])
            for key in turn_batch:
                if key in field:
                    entry[key] = json.dumps(turn_batch[key][i])
                else:
                    pass #ndarray
            results.append(entry)
        write_header = False
        if not self.result_file:
            self.result_file = open(self.cfg.result_path, 'w')
            self.result_file.write(str(self.cfg))
            write_header = True

        writer = csv.DictWriter(self.result_file, fieldnames=field)
        if write_header:
            self.result_file.write('START_CSV_SECTION\n')
            writer.writeheader()
        writer.writerows(results)
        return results


def pad_sequences(sequences, maxlen=None, dtype='int32',
                  padding='pre', truncating='pre', value=0.):
    if not hasattr(sequences, '__len__'):
        raise ValueError('`sequences` must be iterable.')
    lengths = []
    for x in sequences:
        if not hasattr(x, '__len__'):
            raise ValueError('`sequences` must be a list of iterables. '
                             'Found non-iterable: ' + str(x))
        lengths.append(len(x))

    num_samples = len(sequences)
    seq_maxlen = np.max(lengths)
    if maxlen is not None:
        maxlen = min(seq_maxlen, maxlen)
    else:
        maxlen = seq_maxlen
    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((num_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue  # empty list/array was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x
