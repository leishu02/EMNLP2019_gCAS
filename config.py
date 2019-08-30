import logging
import time
import domain_config


class Config:
    def __init__(self, domain):
        self._init_logging_handler()
        self.network = None
        self.cuda_device = 0
        self.seed = 0
        self.spv_proportion = 100
        self.truncated = False

        self.domain = domain
        self.dialog_path = domain_config.domain_path[domain]['dialog_path']
        self.act_path = domain_config.domain_path[domain]['act_path']
        self.slot_path = domain_config.domain_path[domain]['slot_path']
        self.split = domain_config.domain_path[domain]['split']

    def init_handler(self, network_type):
        self.network = network_type
        init_method = {
            'gcas': self._gcas_init,
        }
        init_method[network_type]()

    def _gcas_init(self):
        self.grad_clip_norm = 10.0
        self.max_turn = 100
        self.emb_size = 64
        self.hidden_size = 64
        self.lr = 0.001
        self.lr_decay = 1.0
        self.batch_size = 32
        self.dropout_rate = 0.0
        self.epoch_num = 100  # triggered by early stop
        self.cuda = False #use GPU or not
        self.early_stop_count = 30
        self.vocab_size = None
        self.remove_slot_value = True
        self.encoder_layer_num = 1 #the number of layer of encoder
        self.model_path = './models/gcas_'+self.domain+'.pkl'
        self.result_path = './results/gcas_' + self.domain + '.csv'
        self.teacher_force = 50 #0~99
        if self.remove_slot_value:
            self.vocab_path = './vocabs/' + self.domain + '_woValue.p'
        else:
            self.vocab_path = './vocabs/' + self.domain + '_wValue.p'
        self.continue_size = 4
        if self.domain == 'movie':
            self.user_max_ts = 49
            self.current_slot_max_ts = 82
            self.current_singleslot_max_ts = 36
            self.agent_max_ts = 72
            self.kb_turn_size = 161
            self.cas_max_ts = 6
            self.act_size = 13
            self.slot_size = 58
        elif self.domain == 'restaurant':
            self.user_max_ts = 53
            self.current_slot_max_ts = 71
            self.current_singleslot_max_ts = 34
            self.agent_max_ts = 48
            self.kb_turn_size = 165
            self.cas_max_ts = 5
            self.act_size = 13
            self.slot_size = 62
        elif self.domain == 'taxi':
            self.user_max_ts = 47
            self.current_slot_max_ts = 67
            self.current_singleslot_max_ts = 32
            self.agent_max_ts = 33
            self.kb_turn_size = 147
            self.cas_max_ts = 4
            self.act_size = 13
            self.slot_size = 42
        self.loss_weights = [1.0, 1.0, float(self.slot_size)]#weights for Continue loss, Act loss and Slots loss

    def __str__(self):
        s = ''
        for k, v in self.__dict__.items():
            s += '{} : {}\n'.format(k, v)
        return s

    def _init_logging_handler(self):
        current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        stderr_handler = logging.StreamHandler()
        #file_handler = logging.FileHandler('./log/log_{}.txt'.format(current_time))
        logging.basicConfig(handlers=[stderr_handler])#, file_handler
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)


