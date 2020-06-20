import argparse
import torch
from trainer.meta import MetaTrainer
from trainer.pre import PreTrainer
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', type=str, default='MiniImageNet', choices=['MiniImageNet', 'FC100'])
    parser.add_argument('--phase', type=str, default='meta_train', choices=['pre_train', 'meta_train', 'meta_eval'])
    parser.add_argument('--data_dir', type=str, default='./data/MiniImageNet/')
    parser.add_argument('--gpu', default='1') # GPU id

    parser.add_argument('--epoch', type=int, default=100) # Epoch number for meta-train phase
    parser.add_argument('--batch', type=int, default=100) # The number for different tasks used for meta-train
    parser.add_argument('--shot', type=int, default=1) # Shot number, how many samples for one class in a task
    parser.add_argument('--way', type=int, defalut=5) # Way number, how many classes in a task
    parser.add_argument('--train_query', type=int, default=15) # The number of training samples for each class in a task
    parser.add_argument('--val_query', type=int, default=15) # The number of test samples for each class in a task

    parser.add_argument('--init_weights', type=str, default=None) # The pre-trained weights for meta-train phase
    parser.add_argument('--eval_weights', type=int, default=None) # The meta-trained weights for meta-eval phase

    parser.add_argument('--pre_epoch', type=int, default=100)
    parser.add_argument('--pre_batch', type=int, default=128)

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    torch.backends.cudnn.benchmark = True

    if args.phase == 'meta_train':
        trainer = MetaTrainer(args)
        trainer.train()
    elif args.phase == 'meta_eval':
        trainer = MetaTrainer(args)
        trainer.eval()
    elif args.phase == 'pre_train':
        trainer = PreTrainer(args)
        trainer.train()
    else:
        raise ValueError('Set correct phase')