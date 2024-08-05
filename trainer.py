import os
import torch

from custom_seg_model import CustomSegModel

class Trainer:
    def __init__(self, CONFIG, train_loader, val_loader):
        self.cur_step = 0
        self.config = CONFIG
        self.device = 'cpu' if not torch.cuda.is_available() else self.config.device
        self.model = self.init_model(CONFIG).to(self.device)
        self.train_data, self.val_data = train_loader, val_loader
    
    def init_model(self, config):
        if config.pretrained:
            path = os.path.join(config.ckpt_dir, config.pretrained)
            saved = torch.load(path, map_location='cpu')
            model_config = saved['config']
            model = CustomSegModel(model_config)
            model.model.load_state_dict(saved['encoder_state_dict'])
            model.decode_head.load_state_dict(saved['decoder_state_dict'])
#             model = nn.DataParallel(model)
        else:
            model = CustomSegModel(config)
        return model

    def fit(self, epochs=5):
        losses = []
        # self.model.eval_one_epoch
        for i in range(epochs):
            self.model.train_one_epoch(self.train_data, i)
            self.model.eval_one_epoch(self.val_data, i)
