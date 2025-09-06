import argparse


def get_config():
    parser = argparse.ArgumentParser(description="HIST")

    parser.add_argument("--dataset", type=str, default="coco", help="cifar/coco/...")
    parser.add_argument("--backbone", type=str, default="resnet50", help="resnet50")
    parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs to train for")
    parser.add_argument("--n_workers", type=int, default=4, help="number of dataloader workers")
    parser.add_argument("--batch_size", type=int, default=32, help="input batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="weight decay")
    parser.add_argument("--data_dir", default="../_datasets", help="directory to dataset")
    parser.add_argument("--save_dir", default="./output", help="directory to output results")
    parser.add_argument("--n_classes", type=int, default=10, help="number of dataset classes")
    parser.add_argument("--n_bits", type=int, default=16, help="length of hashing binary")
    parser.add_argument("--topk", type=int, default=5000, help="mAP@topk")

    parser.add_argument("--freeze_BN", default=True, type=bool, help="freeze batch normalization parameter?")
    parser.add_argument("--use_LN", default=True, type=bool, help="use layer normalization?")
    parser.add_argument("--add_GMP", default=True, type=bool, help="GAP+GMP? or only use GAP")

    parser.add_argument("--dim_hidden", default=512, type=int, help="size of hidden dimension in HGNN")

    parser.add_argument("--lr_ds", type=float, default=1e-1, help="lr for class prototypical distribution parameters")
    parser.add_argument("--lr_hgnn_factor", type=float, default=10, help="lr multiplication factor for HGNN parameters")

    parser.add_argument("--alpha", default=0.9, type=float, help="hardness scale parameter for construction of H")
    parser.add_argument("--tau", default=32, type=float, help="temperature scale parameter for softmax")
    parser.add_argument("--ls", default=1, type=float, help="loss scale balancing parameters (lambda_s)")

    parser.add_argument("--lr-decay-step", default=10, type=int, help="learning decay step setting")

    parser.add_argument("--lr-decay-gamma", default=0.5, type=float, help="learning decay gamma setting")

    return parser.parse_args()
