import json
import os
import time
from copy import deepcopy

import torch
import torch.optim as optim
from loguru import logger
from timm.utils import AverageMeter

from _data import build_loader, get_topk, get_class_num
from _utils import prediction, mean_average_precision, calc_net_params, init
from config import get_config
from loss import AsymmetricLoss
from HIST.network import build_model


def train_epoch(args, dataloader, net, criterion, optimizer, epoch):
    tic = time.time()
    loss_meter = AverageMeter()
    map_meter = AverageMeter()
    net.train()
    for images, labels, _ in dataloader:
        images, labels = images.cuda(), labels.cuda()
        logits = net(images)

        loss = criterion(logits, labels)
        loss_meter.update(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # to check overfitting
        q_cnt = labels.shape[0] // 10
        map_k = mean_average_precision(logits[:q_cnt], logits[q_cnt:], labels[:q_cnt], labels[q_cnt:], args.topk)
        map_meter.update(map_k)

    toc = time.time()
    logger.info(
        f"[Training][dataset:{args.dataset}][bits:{args.n_bits}][epoch:{epoch}/{args.n_epochs - 1}][time:{(toc - tic):.3f}][loss:{loss_meter.avg:.4f}][mAP@{args.topk}:{map_meter.avg:.4f}]"
    )


def train_val(args, train_loader, query_loader, dbase_loader, logger):
    # setup net
    net = build_model(args, True)
    logger.info(f"number of net's params: {calc_net_params(net)}")

    # setup criterion
    criterion = AsymmetricLoss(
        gamma_neg=args.gamma_neg,
        gamma_pos=args.gamma_pos,
        clip=args.clip,
        disable_torch_grad_focal_loss=args.disable_torch_grad_focal_loss,
    )

    # setup optimizer
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # training process
    best_map = 0.0
    best_epoch = 0
    best_checkpoint = None
    count = 0
    for epoch in range(args.n_epochs):
        train_epoch(args, train_loader, net, criterion, optimizer, epoch)

        # we monitor MAP@R validation accuracy every 5 epochs
        # and use early stopping patience of 10 to get the best params of model
        if (epoch + 1) % 5 == 0 or (epoch + 1) == args.n_epochs:
            qB, qL = prediction(net, query_loader)
            rB, rL = prediction(net, dbase_loader)
            map_k = mean_average_precision(qB, rB, qL, rL, args.topk)
            del qB, qL, rB, rL
            logger.info(
                f"[Evaluating][dataset:{args.dataset}][bits:{args.n_bits}][epoch:{epoch}/{args.n_epochs - 1}][best-mAP@{args.topk}:{best_map:.4f}][mAP@{args.topk}:{map_k:.4f}][count:{0 if map_k > best_map else (count + 1)}]"
            )

            if map_k > best_map:
                best_map = map_k
                best_epoch = epoch
                best_checkpoint = deepcopy(net.state_dict())
                count = 0
            else:
                count += 1
                # if count == 5:
                #     if criterion.change_hardness():
                #         count = 0
                if count == 20:
                    logger.info(
                        f"without improvement, will save & exit, best mAP: {best_map}, best epoch: {best_epoch}"
                    )
                    torch.save(best_checkpoint, f"{args.save_dir}/e{best_epoch}_{best_map:.3f}.pkl")
                    break
    if count != 10:
        logger.info(f"reach epoch limit, will save & exit, best mAP: {best_map}, best epoch: {best_epoch}")
        torch.save(best_checkpoint, f"{args.save_dir}/e{best_epoch}_{best_map:.3f}.pkl")

    return best_epoch, best_map


def prepare_loaders(args, bl_func):
    train_loader, query_loader, dbase_loader = (
        bl_func(
            args.data_dir,
            args.dataset,
            "train",
            None,
            batch_size=args.batch_size,
            num_workers=args.n_workers,
            drop_last=True,
        ),
        bl_func(args.data_dir, args.dataset, "query", None, batch_size=args.batch_size, num_workers=args.n_workers),
        bl_func(args.data_dir, args.dataset, "dbase", None, batch_size=args.batch_size, num_workers=args.n_workers),
    )
    return train_loader, query_loader, dbase_loader


def main():

    init("0", 0)

    args = get_config()

    dummy_logger_id = None
    rst = []
    for dataset in ["coco", "cifar", "flickr", "nuswide"]:
        print(f"processing dataset: {dataset}")
        args.dataset = dataset
        args.n_classes = get_class_num(dataset)
        args.topk = get_topk(dataset)

        train_loader, query_loader, dbase_loader = prepare_loaders(args, build_loader)

        # for hash_bit in [16, 32, 64, 128]:
        for hash_bit in [16,32,64,128]:
            print(f"processing hash-bit: {hash_bit}")
            args.n_bits = hash_bit

            args.save_dir = f"./output/{args.backbone}/{dataset}/{hash_bit}"
            os.makedirs(args.save_dir, exist_ok=True)
            if any(fname.endswith(".pkl") for fname in os.listdir(args.save_dir)):
                raise Exception(f"*.pkl exists in {args.save_dir}")

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
