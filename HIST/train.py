import json
import os
import time
from copy import deepcopy

import torch
import torch.optim as optim
from loguru import logger
from timm.utils import AverageMeter
from torchvision import transforms

from _data import build_loader, get_topk, get_class_num
from _utils import prediction, init, mean_average_precision, calc_net_params
from config import get_config
from loss import PD4HG
from network import build_model, HGNN


def train_epoch(args, dataloader, net_cnn, net_hgnn, criterion, optimizer, epoch):
    tic = time.time()
    loss_meters = {}
    for x in ["dist", "ce", "all"]:
        loss_meters[x] = AverageMeter()
    map_meter = AverageMeter()

    net_cnn.train()
    for images, labels, _ in dataloader:
        images, labels = images.cuda(), labels.cuda()
        z = net_cnn(images)

        # hypergraph construction & distribution loss
        dist_loss, S = criterion(z, labels)
        loss_meters["dist"].update(dist_loss.item())

        # hypergraph node classification
        out = net_hgnn(z, S)
        ce_loss = torch.nn.CrossEntropyLoss()(out, labels)
        loss_meters["ce"].update(ce_loss.item())

        loss = dist_loss + args.ls * ce_loss
        loss_meters["all"].update(loss.item())

        if torch.isnan(loss):
            if torch.isnan(dist_loss):
                logger.warning("NaN in dist_loss")
            if torch.isnan(ce_loss):
                logger.warning("NaN in ce_loss")
            exit(-1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # to check overfitting
        q_cnt = labels.shape[0] // 10
        map_k = mean_average_precision(z[:q_cnt], z[q_cnt:], labels[:q_cnt], labels[q_cnt:], args.topk)
        map_meter.update(map_k)

    toc = time.time()
    loss_str = ""
    for x in loss_meters.keys():
        loss_str += f"[{x}-loss:{loss_meters[x].avg:.4f}]"
    logger.info(
        f"[Training][dataset:{args.dataset}][bits:{args.n_bits}][epoch:{epoch}/{args.n_epochs - 1}][time:{(toc - tic):.3f}]{loss_str}[mAP@{args.topk}:{map_meter.avg:.4f}]"
    )


def train_val(args, train_loader, query_loader, dbase_loader, logger):
    # setup net
    net_cnn = build_model(args, pretrained=True)

    net_hgnn = HGNN(args.n_classes, args.n_bits, args.dim_hidden).cuda()

    # setup criterion
    criterion = PD4HG(args.n_classes, args.n_bits, args.tau, args.alpha).cuda()

    logger.info(f"number of net's params: {calc_net_params(net_cnn, net_hgnn, criterion)}")

    # setup optimizer
    params = [
        {"params": net_cnn.parameters(), "lr": args.lr},
        {"params": net_hgnn.parameters(), "lr": args.lr * args.lr_hgnn_factor},
        {"params": criterion.parameters(), "lr": args.lr_ds},
    ]
    optimizer = optim.Adam(params, weight_decay=args.weight_decay)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay_gamma)

    # training process
    best_map = 0.0
    best_epoch = 0
    best_checkpoint = None
    count = 0
    for epoch in range(args.n_epochs):
        train_epoch(args, train_loader, net_cnn, net_hgnn, criterion, optimizer, epoch)
        scheduler.step()
        # until convergence
        if (epoch + 1) % 5 == 0 or (epoch + 1) == args.n_epochs:
            qB, qL = prediction(net_cnn, query_loader)
            rB, rL = prediction(net_cnn, dbase_loader)
            map_k = mean_average_precision(qB, rB, qL, rL, args.topk)
            del qB, qL, rB, rL
            logger.info(
                f"[Evaluating][dataset:{args.dataset}][bits:{args.n_bits}][epoch:{epoch}/{args.n_epochs - 1}][best-mAP@{args.topk}:{best_map}][mAP@{args.topk}:{map_k}][count:{0 if map_k > best_map else (count + 1)}]"
            )

            if map_k > best_map:
                best_map = map_k
                best_epoch = epoch
                best_checkpoint = deepcopy(net_cnn.state_dict())
                count = 0
            else:
                count += 1
                if count == 10:
                    logger.info(
                        f"without improvement, will save & exit, best mAP: {best_map}, best epoch: {best_epoch}"
                    )
                    torch.save(best_checkpoint, f"{args.save_dir}/e{best_epoch}_{best_map:.3f}.pkl")
                    break
    if count != 10:
        logger.info(f"reach epoch limit, will save & exit, best mAP: {best_map}, best epoch: {best_epoch}")
        torch.save(best_checkpoint, f"{args.save_dir}/e{best_epoch}_{best_map:.3f}.pkl")

    return best_epoch, best_map


def build_trans(is_train=True):
    resnet_sz_resize = 256
    resnet_sz_crop = 224
    resnet_mean = [0.485, 0.456, 0.406]
    resnet_std = [0.229, 0.224, 0.225]
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(resnet_sz_crop) if is_train else torch.nn.Identity(),
            transforms.RandomHorizontalFlip() if is_train else torch.nn.Identity(),
            transforms.Resize(resnet_sz_resize),
            transforms.CenterCrop(resnet_sz_crop),
            transforms.ToTensor(),
            transforms.Normalize(mean=resnet_mean, std=resnet_std),
        ]
    )


def prepare_loaders(args, bl_func):
    train_loader, query_loader, dbase_loader = (
        bl_func(
            args.data_dir,
            args.dataset,
            "train",
            build_trans(),
            batch_size=args.batch_size,
            num_workers=args.n_workers,
            drop_last=True,
        ),
        bl_func(
            args.data_dir,
            args.dataset,
            "query",
            build_trans(False),
            batch_size=args.batch_size,
            num_workers=args.n_workers,
        ),
        bl_func(
            args.data_dir,
            args.dataset,
            "dbase",
            build_trans(False),
            batch_size=args.batch_size,
            num_workers=args.n_workers,
        ),
    )
    return train_loader, query_loader, dbase_loader


def main():
    init("0", 0)

    args = get_config()

    dummy_logger_id = None
    rst = []
    # for dataset in ["cifar", "nuswide", "flickr", "coco"]:
    for dataset in ["nuswide"]:
        print(f"processing dataset: {dataset}")
        args.dataset = dataset
        args.n_classes = get_class_num(dataset)
        args.topk = get_topk(dataset)

        train_loader, query_loader, dbase_loader = prepare_loaders(args, build_loader)

        # for hash_bit in [16, 32, 64, 128]:
        for hash_bit in [32]:
            print(f"processing hash-bit: {hash_bit}")
            args.n_bits = hash_bit

            args.save_dir = f"./output/{args.backbone}/{dataset}/{hash_bit}"
            os.makedirs(args.save_dir, exist_ok=False)

            if dummy_logger_id is not None:
                logger.remove(dummy_logger_id)
            dummy_logger_id = logger.add(f"{args.save_dir}/train.log", rotation="500 MB", level="INFO")

            with open(f"{args.save_dir}/config.json", "w+") as f:
                json.dump(vars(args), f, indent=4, sort_keys=True)

            best_epoch, best_map = train_val(args, train_loader, query_loader, dbase_loader, logger)
            rst.append({"dataset": dataset, "hash_bit": hash_bit, "best_epoch": best_epoch, "best_map": best_map})
    for x in rst:
        print(
            f"[dataset:{x['dataset']}][bits:{x['hash_bit']}][best-epoch:{x['best_epoch']}][best-mAP:{x['best_map']:.3f}]"
        )


if __name__ == "__main__":
    main()
