import argparse
import torch
import os
import json
import pickle
from config import config
from data_load import data_load
from models.transformer import TransformerCap
from utils.metric import coco_eval
from utils.search import beam_search
from utils.vocab import Vocabulary

# ******************************
#       evaluation prepare
# ******************************

# make log direction
log_path = config.log_dir.format(config.id)

# load vocabulary
with open(config.vocab, 'rb') as f:
    vocab = pickle.load(f)

# load data
loader = data_load(config, 'test')

# load model
model = TransformerCap(config).to('cuda')
ckpts_path = os.path.join(log_path, 'model')
ckpts = os.listdir(ckpts_path)
# scores = [[float(c.split('_')[-1][:-3]), c] for c in ckpts]
# scores.sort(key=lambda x:x[0], reverse=True)
# for c in scores[5:]:
#     os.remove(os.path.join(ckpts_path, c[1]))

score_saves = {}
scores = [[float(c.split('_')[-1][:-3]), c] for c in ckpts]
scores.sort(key=lambda x:x[0], reverse=True)
ckpts = [c[1] for c in scores]
ckpts = ckpts[0] if config.eval == 'best' else ckpts
for c in ckpts:
    ckpt = torch.load(os.path.join(ckpts_path, c))
    model.load_state_dict(ckpt)

    # ******************************
    #           validation
    # ******************************

    model.eval()
    res = []
    with torch.no_grad():
        for step, data in enumerate(loader):
            img_id = data['img_id']
            feat = data['feat'].to('cuda')
            bs = len(feat)
            memory = model.encode(feat)
            sent = beam_search(model.decode, config.num_beams, memory)

            for i, img_id in enumerate(img_id):
                s = vocab.idList_to_sent(sent[i].to('cpu'))
                res.append({'image_id': int(img_id), 'caption': s})
                print(f"{int(img_id)}: {s}")

        # save generated sentence
        res_path = os.path.join(log_path, 'test_result.json')
        with open(res_path, 'w') as f:
            json.dump(res, f)

        # coco evaluation
        scores = coco_eval(config.val_gts, res_path)

        item = {m:s for m, s in scores.items()}
        score_saves[c] = item


# print results
for c, v in score_saves.items():
    print('-'*50)
    print(c)
    for m, s in v.items():
        print(m, s)