import cPickle as pickle
import csv
import json
import numpy as np

def match(gt_list, pred_list):
    gt_matched = [0 for i in range(len(gt_list))]
    pred_matched = [0 for i in range(len(pred_list))]
    for i, g in enumerate(gt_list):
        for j, p in enumerate(pred_list):
            if g == p:
                gt_matched[i] = 1
                pred_matched[j] = 1
    return gt_matched, pred_matched

class Evaluator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.file = open(cfg.result_path, 'r')
        self.meta = []
        self.metric_dict = {}
        filename = cfg.result_path.split('/')[-1]
        dump_dir = './sheets/' + filename.replace('.csv', '.report.txt')
        self.dump_file = open(dump_dir, 'w')

    def run_metrics(self):
        data = self.read_result_data()
        precision, recall, fscore = self.prf_act_slot_pairs(data)
        jaccard = self.turn_jaccard_index(data)
        act_precision, act_recall, act_fscore = self.prf_act_slot_pairs(data, mode='act')
        act_jaccard = self.turn_jaccard_index(data, mode='act')
        success_p, success_r, success_f = self.success_f1_metric(data)
        entity_p, entity_r, entity_f = self.entity_f1_metric(data)
        inform_all_p, inform_all_r, inform_all_f = self.prf_inform_slot_pairs(data, mode='all')
        inform_critical_p, inform_critical_r, inform_critical_f = self.prf_inform_slot_pairs(data, mode='critical')
        inform_noncritical_p, inform_noncritical_r, inform_noncritical_f = self.prf_inform_slot_pairs(data, mode='noncritical')
        self.metric_dict['pair_precision'] = precision
        self.metric_dict['pair_recall'] = recall
        self.metric_dict['pair_fscore'] = fscore
        self.metric_dict['pair_jaccard'] = jaccard
        self.metric_dict['act_precision'] = act_precision
        self.metric_dict['act_recall'] = act_recall
        self.metric_dict['act_fscore'] = act_fscore
        self.metric_dict['act_jaccard'] = act_jaccard
        self.metric_dict['success_precision'] = success_p
        self.metric_dict['success_recall'] = success_r
        self.metric_dict['success_fscore'] = success_f
        self.metric_dict['entity_precision'] = entity_p
        self.metric_dict['entity_recall'] = entity_r
        self.metric_dict['entity_fscore'] = entity_f
        self.metric_dict['inform_all_p'] = inform_all_p
        self.metric_dict['inform_all_r'] = inform_all_r
        self.metric_dict['inform_all_f'] = inform_all_f
        self.metric_dict['inform_critical_p'] = inform_critical_p
        self.metric_dict['inform_critical_r'] = inform_critical_r
        self.metric_dict['inform_critical_f'] = inform_critical_f
        self.metric_dict['inform_noncritical_p'] = inform_noncritical_p
        self.metric_dict['inform_noncritical_r'] = inform_noncritical_r
        self.metric_dict['inform_noncritical_f'] = inform_noncritical_f
        self.dump()
        return precision, recall, fscore

    def retrieve_act_list(self, act_slot_pairs):
        output = []
        for pair in act_slot_pairs:
            k, v = pair.split('+')
            output.append(k)
        return output

    def retrieve_slot(self, act_slot_pair):
        k, v = act_slot_pair.split('+')
        return v

    def success_f1_metric(self, data):#check if agent inform user's request
        dials = self.pack_dial(data)
        tp, fp, fn = 0, 0, 0
        for dial_id in dials:
            user_req, truth_pair, gen_pair = [], [], []
            dial = dials[dial_id]
            for turn_num, turn in enumerate(dial):
                request_slots = json.loads(turn['state'])['current_slots']['request_slots']
                gt_y = json.loads(turn['act_slot_pairs'])
                pred_y = json.loads(turn['pred_act_slot_pairs'])
                user_req += request_slots.keys()
                truth_pair += [y.replace('inform+','') for y in gt_y if 'inform' in y]
                gen_pair += [y.replace('inform+', '') for y in pred_y if 'inform' in y]
            user_req = set(user_req)
            truth_pair = set(truth_pair)
            gen_pair = set(gen_pair)
            cleaned_truth_pair = [p for p in truth_pair if p in user_req or p == 'taskcomplete' or p== 'result' or p=='mc_list']
            cleaned_gen_pair = [p for p in gen_pair if p in user_req or p == 'taskcomplete' or p == 'result' or p=='mc_list']
            for req in cleaned_gen_pair:
                if req in cleaned_truth_pair:
                    tp += 1
                else:
                    fp += 1
            for req in cleaned_truth_pair:
                if req not in cleaned_gen_pair:
                    fn += 1
        precision, recall = tp / (tp + fp + 1e-8), tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        return precision, recall, f1

    def entity_f1_metric(self, data):#check if agent inform user's request
        dials = self.pack_dial(data)
        tp, fp, fn = 0, 0, 0
        for dial_id in dials:
            user_req, truth_pair, gen_pair = [], [], []
            dial = dials[dial_id]
            user_inf = json.loads(dial[-1]['state'])['current_slots']['inform_slots'].keys()
            for turn_num, turn in enumerate(dial):
                gt_y = json.loads(turn['act_slot_pairs'])
                pred_y = json.loads(turn['pred_act_slot_pairs'])
                truth_pair += [self.retrieve_slot(y) for y in gt_y if 'inform' not in y]
                gen_pair += [self.retrieve_slot(y) for y in pred_y if 'inform' not in y]
            truth_pair = set(truth_pair)
            gen_pair = set(gen_pair)
            cleaned_truth_pair = [p for p in truth_pair if p in user_inf and p != 'other']
            cleaned_gen_pair = [p for p in gen_pair if p in user_inf and p != 'other']
            for req in cleaned_gen_pair:
                if req in cleaned_truth_pair:
                    tp += 1
                else:
                    fp += 1
            for req in cleaned_truth_pair:
                if req not in cleaned_gen_pair:
                    fn += 1
        precision, recall = tp / (tp + fp + 1e-8), tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        return precision, recall, f1

    def prf_act_slot_pairs(self, data, mode='pair'):

        dials = self.pack_dial(data)
        # find out the last placeholder and see whether that is correct
        # if no such placeholder, see the final turn, because it can be a yes/no question or scheduling dialogue
        total_gt_matched = []
        total_pred_matched = []
        for dial_id in dials:
            dial = dials[dial_id]
            for turn_num, turn in enumerate(dial):
                gt_y = json.loads(turn['act_slot_pairs'])
                pred_y = json.loads(turn['pred_act_slot_pairs'])
                if mode == 'act':
                    gt_y = self.retrieve_act_list(gt_y)
                    pred_y = self.retrieve_act_list(pred_y)
                gt_matched, pred_matched = match(gt_y, pred_y)
                total_gt_matched += gt_matched
                total_pred_matched += pred_matched
        precision = float(sum(total_pred_matched)) / (float(len(total_pred_matched)) + 1.e-8)
        recall = float(sum(total_gt_matched)) / (float(len(total_gt_matched)) + 1.e-8)
        fscore = 2 * precision * recall / (precision + recall + 1.e-8)
        return precision, recall, fscore


    def prf_inform_slot_pairs(self, data, mode='all'):
        def pick_inform(pairs, user_inf):
            output = []
            for pair in pairs:
                k, v = pair.split('+')
                if k == 'inform' and v not in ['', 'other']:
                    if mode == 'all':
                        output.append(v)
                    elif mode == 'critical':
                        if v not in user_inf:
                            output.append(v)
                    elif mode == 'noncritical':
                        if v in user_inf:
                            output.append(v)
                    else:
                        assert()
            return list(set(output))

        dials = self.pack_dial(data)
        # find out the last placeholder and see whether that is correct
        # if no such placeholder, see the final turn, because it can be a yes/no question or scheduling dialogue
        total_gt_matched = []
        total_pred_matched = []
        for dial_id in dials:
            dial = dials[dial_id]
            for turn_num, turn in enumerate(dial):
                user_inf = json.loads(turn['state'])['current_slots']['inform_slots'].keys()
                gt_y = pick_inform(json.loads(turn['act_slot_pairs']), user_inf)
                pred_y = pick_inform(json.loads(turn['pred_act_slot_pairs']), user_inf)
                gt_matched, pred_matched = match(gt_y, pred_y)
                total_gt_matched += gt_matched
                total_pred_matched += pred_matched
        precision = float(sum(total_pred_matched)) / (float(len(total_pred_matched)) + 1.e-8)
        recall = float(sum(total_gt_matched)) / (float(len(total_gt_matched)) + 1.e-8)
        fscore = 2 * precision * recall / (precision + recall + 1.e-8)
        return precision, recall, fscore

    def turn_jaccard_index(self, data, mode='pair'):
        dials = self.pack_dial(data)
        # find out the last placeholder and see whether that is correct
        # if no such placeholder, see the final turn, because it can be a yes/no question or scheduling dialogue
        total_score = []
        for dial_id in dials:
            dial = dials[dial_id]
            for turn_num, turn in enumerate(dial):
                gt_y = json.loads(turn['act_slot_pairs'])
                pred_y = json.loads(turn['pred_act_slot_pairs'])
                if mode == 'act':
                    gt_y = self.retrieve_act_list(gt_y)
                    pred_y = self.retrieve_act_list(pred_y)
                intersection = set(gt_y).intersection(set(pred_y))
                union = set().union(gt_y, pred_y)
                score = float(len(intersection) + 1.e-8) / (float(len(union)) + 1.e-8)
                total_score.append(score)

        return np.mean(total_score)

    def turn_prf_act_slot_pairs(self, data, mode='pair'):
        dials = self.pack_dial(data)
        # find out the last placeholder and see whether that is correct
        # if no such placeholder, see the final turn, because it can be a yes/no question or scheduling dialogue
        total_precision = []
        total_recall = []
        total_fscore = []
        for dial_id in dials:
            dial = dials[dial_id]
            for turn_num, turn in enumerate(dial):
                gt_y = json.loads(turn['act_slot_pairs'])
                pred_y = json.loads(turn['pred_act_slot_pairs'])
                if mode == 'act':
                    gt_y = self.retrieve_act_list(gt_y)
                    pred_y = self.retrieve_act_list(pred_y)
                gt_matched, pred_matched = match(gt_y, pred_y)
                precision = float(sum(pred_matched) + 1.e-8) / (float(len(pred_matched)) + 1.e-8)
                recall = float(sum(gt_matched) + 1.e-8) / (float(len(gt_matched)) + 1.e-8)
                fscore = 2 * precision * recall / (precision + recall + 1.e-8)
                total_precision.append(precision)
                total_recall.append(recall)
                total_fscore.append(fscore)

        return np.mean(total_precision), np.mean(total_recall), np.mean(total_fscore)

    def read_result_data(self):
        while True:
            line = self.file.readline()
            if 'START_CSV_SECTION' in line:
                break
        self.meta.append(line)
        reader = csv.DictReader(self.file)
        data = [_ for _ in reader]
        return data

    def pack_dial(self, data):
        dials = {}
        for turn in data:
            dial_id = int(turn['dial_id'])
            if dial_id not in dials:
                dials[dial_id] = []
            dials[dial_id].append(turn)
        return dials

    def dump(self):
        #self.dump_file.writelines(self.meta)
        self.dump_file.write('START_REPORT_SECTION\n')
        for k, v in self.metric_dict.items():
            self.dump_file.write('{}\t{}\n'.format(k, v))

