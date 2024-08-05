import torch

from transformers import SegformerConfig, Trainer
from custom_seg_model import CustomSegModel
from dataset import SlidingWindowDataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split

CFG = SegformerConfig.from_pretrained("nvidia/segformer-b5-finetuned-cityscapes-1024-1024")
CFG.pretrained = None
CFG.from_zoo = None
CFG.num_labels = 2
CFG.num_channels = 18
CFG.ckpt_dir="model/checkpoints"

CFG.lr = 1e-3

CFG.src_dir = "SRC"
CFG.batch_size = 2
CFG.train_ratio = 0.9
CFG.num_workers = 2
CFG.accumulate = 4

CFG.device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = CustomSegModel(CFG)

train_val_split = (0.9, 0.1)

dataset = SlidingWindowDataset(pickle_dir="data/train/training_2015_pickled_data", window_size=128, stride=64, reduce_indices=True)

generator = torch.Generator().manual_seed(123)
train_dataset, val_dataset = random_split(dataset, train_val_split, generator=generator)

train_loader = DataLoader(train_dataset, batch_size=CFG.batch_size, shuffle=True, num_workers=CFG.num_workers)
val_loader = DataLoader(val_dataset, batch_size=CFG.batch_size, shuffle=False, num_workers=CFG.num_workers)

trainer = Trainer(CFG, train_loader, val_loader)

trainer.fit(epochs=5)