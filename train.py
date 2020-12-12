import torch
import random
import numpy as np

import time
import os
import json
import pickle
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from config import config
from data_load import data_load
from models.transformer import TransformerCap
from utils.optimizer import NoamOpt
from utils.loss import LanguageModelCriterion
from utils.save import write_scalar, save_model
from utils.log_print import train_print
from utils.search import beam_search
from utils.metric import coco_eval
from utils.vocab import Vocabulary

# ******************************
#       training prepare
# ******************************

# set fixed seed
seed = config.seed
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True

# make log direction
log_path = config.log_dir.format(config.id)
if not os.path.exists(log_path):
    os.makedirs(log_path)

# gobal information
epochs = config.epoch
global_step = 0
best_score = 0
saves = [[0, 0, '0']] * config.topk
writer = SummaryWriter(log_path)
with open(config.vocab, 'rb') as f:
    vocab = pickle.load(f)

# load data
train_loader = data_load(config, 'train')
val_loader = data_load(config, 'val')

# load model
model = TransformerCap(config).to('cuda')

# optimizer
adam = torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
optimizer = NoamOpt(config, adam)
criterion = LanguageModelCriterion()

# ******************************
#       training or validate
# ******************************

for epoch in range(epochs):

    # ******************************
    #           training
    # ******************************

    model.train()
    total_step = len(train_loader)
    epoch_time = time.time()
    for step, data in enumerate(train_loader):
        global_step += 1
        step_time = time.time()

        # prepare data
        tmp = [data['feat'], data['cap'][:, :-1], data['cap'][:, 1:], data['cap_mask'][:, 1:]]
        tmp = [_ if _ is None else _.cuda() for _ in tmp]
        feat, cap, tgt, tgt_mask = tmp

        # model forward and back word
        logit = model(feat, cap)
        loss = criterion(logit, tgt, tgt_mask)
        model.zero_grad()
        nn.utils.clip_grad_value_(model.parameters(), config.grad_clip)
        loss.backward()
        optimizer.step()

        # save learning rate
        # write_scalar(writer, 'lr', optimizer.rate(), global_step)

        # save loss
        if global_step % config.save_loss_freq == 0:
            write_scalar(writer, 'loss', loss.item(), global_step)
        # print training information
        train_print(loss.item(), step, total_step, epoch, time.time() - step_time, time.time() - epoch_time)
        if global_step % config.save_model_freq == 0 and global_step > config.warmup:

            # ******************************
            #           validation
            # ******************************

            model.eval()
            res = []
            with torch.no_grad():
                for step, data in enumerate(val_loader):
                    img_id = data['img_id']
                    feat = data['feat'].to('cuda')
                    memory = model.encode(feat)
                    sent = beam_search(model.decode, config.num_beams, memory)
                    for i, img_id in enumerate(img_id):
                        s = vocab.idList_to_sent(sent[i])
                        res.append({'image_id': int(img_id), 'caption': s})
                        print(f"{int(img_id)}: {s}")

                # save generated sentence
                res_path = os.path.join(log_path, 'val_result.json')
                with open(res_path, 'w') as f:
                    json.dump(res, f)

                # coco evaluation
                scores = coco_eval(config.val_gts, res_path)
                for method, score in scores.items():
                    write_scalar(writer, method, score, global_step)
                sup_score = scores[config.sup_score]

                # save the best model on supervised score
                best_score = save_model(sup_score, global_step, model, saves, log_path)

            model.train()
