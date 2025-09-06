import argparse


def get_config():
    parser = argparse.ArgumentParser(description="ASL")

    parser.add_argument("--dataset", type=str, default="coco", help="/root/autodl-tmp/ASL/_datasets")
    parser.add_argument("--backbone", type=str, default="resnet50", help="see network.py")
    parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs to train for")
    parser.add_argument("--n_workers", type=int, default=4, help="number of dataloader workers")
    parser.add_argument("--batch_size", type=int, default=64, help="input batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="weight decay")
    parser.add_argument("--data_dir", default="/root/autodl-tmp/ASL/_datasets", help="directory to dataset")
    parser.add_argument("--save_dir", default="/root/autodl-tmp/ASL/output", help="directory to output results")
    parser.add_argument("--n_classes", type=int, default=10, help="number of dataset classes")
    parser.add_argument("--n_bits", type=int, default=16, help="length of hashing binary")
    parser.add_argument("--topk", type=int, default=5000, help="mAP@topk")

    parser.add_argument("--gamma_neg", type=int, default=5, help="focusing parameter of negative loss")
    parser.add_argument("--gamma_pos", type=int, default=1, help="focusing parameter of positive loss")
    parser.add_argument("--clip", type=float, default=0.05, help="param for asymmetric clipping (probability shifting)")
    parser.add_argument("--disable_torch_grad_focal_loss", type=bool, default=True, help="")

    parser.add_argument("--freeze_BN", default=True, type=bool, help="freeze batch normalization parameter?")
    parser.add_argument("--use_LN", default=False, type=bool, help="use layer normalization?")
    parser.add_argument("--add_GMP", default=False, type=bool, help="GAP+GMP? or only use GAP")

    return parser.parse_args()
