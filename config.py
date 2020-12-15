import argparse

parser = argparse.ArgumentParser()

# training setting
parser.add_argument('--seed', default=123)
parser.add_argument('--id', required=True, default='baseline')
parser.add_argument('--epoch', default=20)
parser.add_argument('--eval', type=str, default='all', help='best or all')

# load data direction
parser.add_argument('--vocab', default='./data/vocab.pkl')
parser.add_argument('--cap', default='./data/{}.json')
parser.add_argument('--feat', default='./data/cocobu36_att')
parser.add_argument('--val_gts', default='./data/annotations/captions_val2014.json',
                    help='ground truth of coco validation')

# dataloader paramterts
parser.add_argument('--train_bs', default=50)
parser.add_argument('--test_bs', default=100)
parser.add_argument('--num_workers', default=4)

# process caption
parser.add_argument('--fixed_len', default=16)

# save parameters
parser.add_argument('--save_loss_freq', default=50)
parser.add_argument('--save_model_freq', default=3000)
parser.add_argument('--topk', default=5, help='save best model of the first topk ')
parser.add_argument('--log_dir', default='./log/{}')

# supervised metric
parser.add_argument('--sup_score', default='CIDEr')

# model parametsers
parser.add_argument('--num_layers', default=6)
parser.add_argument('--nhead', default=8)
parser.add_argument('--dropout', default=0.1)
parser.add_argument('--d_model', default=512)
parser.add_argument('--d_ff', default=2048)
parser.add_argument('--feat_dim', default=2048)
parser.add_argument('--lm_dropout', default=0.5)
parser.add_argument('--num_beams', default=3)

# optitmizer Noam parameters
parser.add_argument('--factor', type=float, default=1)
parser.add_argument('--warmup', type=int, default=20000)
parser.add_argument('--grad_clip', type=float, default=0.1)

config = parser.parse_args()

# transformer_args = {
#
#     'seed': 123,
#
#     # 读取数据目录
#     'vocab': './data/COCO/vocab.pkl',
#     'text': './data/COCO/{}.json',
#     'bu': './data/COCO/cocobu36_{}',
#     'val_gts': './data/COCO/annotations/captions_val2014.json',
#
#     'train_bs': 10,
#     'test_bs': 100,
#     'num_workers': 4,
#
#     # 词表信息
#     'vocab_size': 9490,
#     'fixed_len': 16,
#
#     # 保存信息
#     'save_loss_freq': 1000,
#     'save_model_freq': 5000,
#     'log_dir': './log/{}',
#
#     # 指标检测
#     'sup_score': 'CIDEr',
#
#     # 模型参数
#     'num_layers': 6,
#     'hid_dim': 2048,
#     'nhead': 8,
#     'dropout': 0.1,
#     'feat_dim': 2048,
#     'd_model': 512,
#     'd_ff': 2048,
#
#
#
# }
#
# config = Namespace(**transformer_args)
