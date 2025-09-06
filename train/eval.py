import os

from config import get_config
from HIST.network import build_model
from _utils import init, init_my_eval

if __name__ == "__main__":

    init("1")

    proj_name = os.path.basename(os.path.dirname(__file__))
    backbone = "resnet50"

    # evals = ["mAP", "NDCG", "PR-curve", "TopN-precision", "P@H≤2", "TrainingTime", "EncodingTime", "SelfCheck"]
    evals = ["mAP", "SelfCheck"]

    datasets = ["cifar", "nuswide", "flickr", "coco"]

    hash_bits = [16, 32, 64, 128]

    init_my_eval(get_config, build_model, None)(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        proj_name,
        backbone,
        evals,
        datasets,
        hash_bits,
        False,
        True,
    )
