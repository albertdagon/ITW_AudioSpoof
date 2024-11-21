import os
import argparse
import json
from pathlib import Path
from shutil import copy
from importlib import import_module

import torch
from torch.utils.tensorboard import SummaryWriter

from utils import set_seed, get_loader, get_model
from evaluation_utils import calculate_tDCF_EER, produce_evaluation_file


def main(args):
    # Parse configuration file
    with open(args.config, "r") as f_json:
        config = json.loads(f_json.read())
    model_config = config["model_config"]
    optim_config = config["optim_config"]
    optim_config["epochs"] = config["num_epochs"]
    track = config["track"]

    if "eval_all_best" not in config:
        config["eval_all_best"] = "True"
    if "freq_aug" not in config:
        config["freq_aug"] = "False"

    # For reproducibility
    set_seed(args.seed, config)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    output_dir = Path(args.output_dir)
    # database_path = Path(config["database_path"])
    database_path = Path("./datasets/LA/")
    dev_trial_path = (
        database_path / "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt"
    )
    eval_trial_path = (
        database_path / "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt"
    )

    # Create file logs folder for checkpoints
    model_logs = "LA_{}_ep{}_bs{}_train".format(
        os.path.splitext(os.path.basename(args.config))[0],
        config["num_epochs"],
        config["batch_size"],
    )
    model_logs = output_dir / model_logs
    model_save_path = model_logs / "weights"
    eval_score_path = model_logs / config["eval_output"]
    writer = SummaryWriter(model_logs)
    os.makedirs(model_save_path, exist_ok=True)
    copy(args.config, model_logs / "config.conf")

    # Define model
    model = get_model(model_config, device)

    # Define train and dev loaders
    _, _, eval_loader = get_loader(database_path, args.seed, config)

    # Define optimizer and scheduler
    if not args.debug:
        model.load_state_dict(torch.load(config["model_path"], map_location=device))
    else:
        dummy_state_dict = {}
        for name, param in model.state_dict().items():
            dummy_state_dict[name] = torch.randn_like(param, dtype=torch.float32)
        model.load_state_dict(dummy_state_dict)
    print("Model loaded : {}".format(config["model_path"]))
    print("Start evaluation...")
    produce_evaluation_file(
        eval_loader, model, device, eval_score_path, eval_trial_path
    )
    calculate_tDCF_EER(
        cm_scores_file=eval_score_path,
        asv_score_file=database_path / config["asv_score_path"],
        output_file=model_logs / "t-DCF_EER.txt",
    )
    print("DONE.")
    eval_eer, eval_tdcf = calculate_tDCF_EER(
        cm_scores_file=eval_score_path,
        asv_score_file=database_path / config["asv_score_path"],
        output_file=model_logs / "loaded_model_t-DCF_EER.txt",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation audio spoofing detection system")
    parser.add_argument(
        "--config", type=str, required=True, help="configuration file for the system"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="output directory for results",
        default="./exp_result",
    )
    parser.add_argument(
        "--seed", type=int, default=40, help="random seed (default: 40)"
    )
    parser.add_argument(
        "--debug", action='store_true', help="debug mode"
    )

    args = parser.parse_args()
    main(args)
