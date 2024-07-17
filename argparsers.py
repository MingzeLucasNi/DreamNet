import argparse
from utilities import *

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cifar100", choices=['cifar100', 'imagenet'])
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--num_workers", type=int, default=10) 
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--mae_type", type=str, default='google')

    args = parser.parse_args()

    num_classes_dict = {'cifar10': 10, 'cifar100': 100, 'imagenet': 1000}
    args.num_classes = num_classes_dict[args.dataset]
    return args