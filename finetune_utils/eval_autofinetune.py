import numpy as np
import finetune
import yaml
from tests.build_parser import build_parser
import shutil
import os
import pandas as pd
from hpo.search_space import  SearchSpace
from meta_album.dataset import AVAILABLE_MTLBM_DATASETS, DEFAULT_FOLDER, AVAILABLE_SETS
import json
import signal


def signal_handler(signum, frame):
    raise Exception("Timed out!")


def time_out_eval(*args, **kwargs):
    time_limit = kwargs.pop("time_limit")
    signal.signal(signal.SIGALRM, signal_handler)
    print("Time limit:", time_limit, " seconds")
    signal.alarm(time_limit)  # Ten seconds
    try:
        perf, cost = eval_finetune_conf(*args, **kwargs)
    except Exception as e:
        print("Timed out!")
        perf = 0
        cost = 0

    return perf, cost

def eval_finetune_conf(conf, task_info, budget=1, experiment="test",
                                                    output= "../experiments/output/temp/",
                                                    data_path="../datasets/meta-album",
                                                    verbose=False,
                                                    dataset_download=False):

    parser = build_parser()
    if "amp" in conf:
        conf.pop("amp")
    bss_reg = conf["bss_reg"]
    cotuning_reg = conf.get("cotuning_reg", 0)
    delta_reg = conf["delta_reg"]
    drop = conf["drop"]
    lr = conf["lr"]
    pct_to_freeze = conf["pct_to_freeze"]
    sp_reg = conf["sp_reg"]
    smoothing = conf["smoothing"]
    warmup_epochs = conf["warmup_epochs"]
    warmup_lr = conf["warmup_lr"]
    weight_decay = conf["weight_decay"]
    mixup = conf["mixup"]
    mixup_prob = conf["mixup_prob"]
    cutmix = conf["cutmix"]
    model = conf["model"]
    batch_size = conf["batch_size"]
    opt = conf["opt"]

    #Task specific arguments
    dataset = task_info["dataset"]
    train_split = task_info["train_split"]
    val_split = task_info["val_split"]
    num_classes = task_info["num_classes"]

    #if dataset.startswith("mtlbm"):
    #    prefix = data_path
    #else:
    #    prefix = ""
    prefix = data_path
    suffix = ""
    if "linear_probing" in conf:
        if conf["linear_probing"]:
            suffix += " --linear_probing"
    if "stoch_norm" in conf:
        if conf["stoch_norm"]:
            suffix += " --stoch_norm"
    if "clip_grad_norm" in conf:
        suffix += " --clip_grad_norm {}".format(conf["clip_grad_norm"])
    if "sched" in conf:
        if conf["sched"] != "None":
            suffix += " --sched {}".format(conf["sched"])
    if "opt_betas" in conf:
        if conf["opt_betas"] != "None":
            conf["opt_betas"] = conf["opt_betas"].replace("[","").replace("]", "").replace(",", "***")
            suffix += " --opt_betas {}".format(conf["opt_betas"])
    if "data_augmentation" in conf:
        suffix += " --data_augmentation {}".format(conf["data_augmentation"])
    if "layer_decay" in conf:
        if conf["layer_decay"] != "None":
            suffix += " --layer_decay {}".format(conf["layer_decay"])
    if "auto_augment" in conf:
        if conf["auto_augment"] != "None":
            suffix += " --auto_augment {}".format(conf["auto_augment"])

    output_dir = os.path.join( output, experiment)
    resume_path = os.path.join(output_dir, "last.pth.tar")
    if os.path.exists(resume_path):
        suffix += f" --resume {resume_path}"
    if dataset_download:
        suffix += " --dataset_download"
    argString = f"{prefix} --batch_size {batch_size} --bss_reg {bss_reg} --cotuning_reg {cotuning_reg} --cutmix {cutmix} --delta_reg {delta_reg} --drop {drop} --lr {lr} --mixup {mixup} --mixup_prob {mixup_prob} --model {model} --opt {opt} --pct_to_freeze {pct_to_freeze} --smoothing {smoothing} --sp_reg {sp_reg} --warmup_epochs {warmup_epochs} --warmup_lr {warmup_lr} --weight_decay {weight_decay} --dataset {dataset} --train-split {train_split} --val-split {val_split} --experiment {experiment} --output {output} --pretrained --checkpoint_hist 1 --num_classes {num_classes} --epochs 50 --epochs_step {budget} --workers 1 " + suffix
    argString = argString.replace("\n", "")
    argString = argString.split(" ")
    argString = [x.replace("***", " ") for x in argString] #recover the space for some arguments
    args, _ = parser.parse_known_args(argString)
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)

    try:
        finetune.main(args, args_text)
    except Exception as e:
        if verbose:
            print("Error:", e)
            print("Args:", args_text)
        return 0, 0, "Error: "+ str(e)

    #read last line of txt
    summary = pd.read_csv(os.path.join(output_dir, "summary.csv"))
    eval_top1 = summary["eval_top1"].iloc[-1]
    eval_time = summary["eval_time"].iloc[-1]

    return float(eval_top1), float(eval_time), "Success"

if __name__ == "__main__":

    opt_code = "random"
    version = "micro"
    set = "set1"
    dataset = AVAILABLE_MTLBM_DATASETS[set][0]

    path = os.path.dirname(__file__)
    with open(os.path.join(path, "..", DEFAULT_FOLDER, version, set, dataset, "info.json")) as f:
        info_json = json.load(f)

    train_split = "train"
    val_split = "val"
    dataset = "mtlbm/{}/{}/{}".format(version, set, dataset)
    num_classes = info_json["total_categories"]

    task_info = {"train_split": train_split,
                 "val_split": val_split,
                 "dataset": dataset,
                 "num_classes": num_classes}

    dataset = dataset.replace("/", "_")
    np.random.seed(0)
    ss = SearchSpace()
    configuration, args = ss.sample_configuration(return_args=True)
    config_code = str(ss.get_configuration_code(configuration))
    configuration = configuration.get_dictionary()
    experiment = f"{config_code}_{dataset}_{opt_code}"
    out = time_out_eval(configuration, task_info, budget=1, experiment=experiment, time_limit=360)
    print(out)
    out = eval_finetune_conf(configuration, task_info, budget=1, experiment=experiment)
    print(out)
    out = eval_finetune_conf(configuration, task_info, budget=2, experiment=experiment)
    print(out)
    out = eval_finetune_conf(configuration, task_info, budget=3, experiment=experiment)
    print(out)

    print("Done")