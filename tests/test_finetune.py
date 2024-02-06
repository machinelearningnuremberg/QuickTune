import numpy as np
import finetune
import yaml
from .build_parser import build_parser
import shutil
import os
import pandas as pd

def test_finetune():
    parser = build_parser()
    argString = "datasets/meta-album --batch_size 32 --bss_reg 0.0001 --cotuning_reg 1 --cutmix 0.25 --delta_reg 0.01 --drop 0.1 --layer_decay 0.65 --linear_probing \
                                     --lr 0.01 --mixup 0.0 --mixup_prob 0.25 --model edgenext_x_small --opt adam --pct_to_freeze 1.0 --smoothing 0.05 --sp_reg 0.01 --stoch_norm \
                                     --warmup_epochs 5 --warmup_lr 1e-05 --weight_decay 1e-05 --auto_augment v0 --opt_betas 0 0.99 --dataset mtlbm/micro/set1/DOG --train-split train \
                                     --val-split val --experiment test_finetune_config --output experiments/output/metrics/ --pretrained --checkpoint_hist 1 --num_classes 19 --epochs 1 --workers 2"
    args, _ = parser.parse_known_args(argString.split(" "))
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    output_dir = os.path.join(args.output, args.experiment)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    finetune.main(args, args_text)

    with open(os.path.join(output_dir, "general_log.yml"), "r") as f:
        info = yaml.safe_load(f)

    assert info["device_count"] == 1, "device_count should be 1"
    assert not info["invalid_loss_value"], "invalid_loss_value should be False"

    df = pd.read_csv(os.path.join(output_dir, "summary.csv"))
    assert df["epoch"].iloc[-1] == 0, "epoch should be 1"
    assert np.isclose(df["train_loss"].iloc[-1], 6.15972, 0.0001), "train_loss should be 6.1597 but is {}".format(df["train_loss"].iloc[-1])
    assert np.isclose(df["train_head_grad_norm"].iloc[-1],  2.24151, 0.0001), "train_head_grad_norm should be 2.2273"
    assert np.isclose(df["train_backbone_grad_norm"].iloc[-1],  0.25700, 0.0001), "train_backbone_grad_norm should be 0.000"
    assert np.isclose(df["eval_loss"].iloc[-1],  3.00238, 0.0001), "eval_loss should be 3.00238"

if __name__ == "__main__":
    test_finetune()

