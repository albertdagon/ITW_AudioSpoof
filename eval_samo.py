import os
import argparse
import json
from pathlib import Path
from shutil import copy
from importlib import import_module

import numpy as np
import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

from utils import set_seed, get_loader, get_model
from evaluation_utils import calculate_tDCF_EER, produce_evaluation_file, compute_eer, calculate_tDCF_EER
from loss import SAMO
from tqdm import tqdm


def main(args):
    # Parse configuration file
    with open(args.config, "r") as f_json:
        config = json.loads(f_json.read())
    model_config = config["model_config"]
    loss_config = config["loss"]
    optim_config = config["optim_config"]
    optim_config["epochs"] = config["num_epochs"]
    track = config["track"]

    if track == "ITW":
        raise ValueError("ITW is only for evaluation")

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
    _, _, eval_data_loader, train_bona_loader, _, eval_enroll_loader, _ = get_loader(database_path, args.seed, config)

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

    model.eval()
    samo = SAMO(loss_config["enc_dim"], m_real=loss_config["m_real"], m_fake=loss_config["m_fake"], alpha=loss_config["alpha"]).to(device)

    with torch.no_grad():
        ip1_loader, utt_loader, idx_loader, score_loader, spk_loader, tag_loader = [], [], [], [], [], []

        eval_enroll = update_embeds(device, model, eval_enroll_loader)
        samo.center = torch.stack(list(eval_enroll.values()))

        for i, (feat, labels, spk, utt, tag) in enumerate(tqdm(eval_data_loader)):
                feat = feat.to(device)
                labels = labels.to(device)
                feats, feat_outputs = model(feat)

                if loss_config["target_only"]:  # loss calculation for target-only speakers
                    _, score = samo(feats, labels, spk, eval_enroll, 1)
                else:
                    _, score = samo.inference(feats, labels, spk, eval_enroll, 1)

                ip1_loader.append(feats)
                idx_loader.append(labels)
                score_loader.append(score)
                utt_loader.extend(utt)
                spk_loader.extend(spk)
                tag_loader.extend(tag)

        scores = torch.cat(score_loader, 0).data.cpu().numpy()
        labels = torch.cat(idx_loader, 0).data.cpu().numpy()
        eer = compute_eer(scores[labels == 0], scores[labels == 1])[0]
    
    with open(eval_score_path, "w") as fh:  # w as in overwrite mode
        for utt, tag, score, label, spk in zip(utt_loader, tag_loader, scores, labels, spk_loader):
            fh.write(f"{utt} {tag} {label} {score} {spk}\n")
    print(f"Scores saved to {eval_score_path}")

    print(f"Test EER: {eer}")
    calculate_tDCF_EER(cm_scores_file=eval_score_path,
        asv_score_file=database_path / config["asv_score_path"],
        output_file=model_logs / "t-DCF_EER.txt")


def update_embeds(device, enroll_model, loader):
    enroll_emb_dict = {}
    with torch.no_grad():
        for i, (batch_x, _, spk, _, _) in enumerate(tqdm(loader)):  # batch_x = x_input, key = utt_list
            batch_x = batch_x.to(device)
            batch_cm_emb, _ = enroll_model(batch_x)
            batch_cm_emb = batch_cm_emb.detach().cpu().numpy()

            for s, cm_emb in zip(spk, batch_cm_emb):
                if s not in enroll_emb_dict:
                    enroll_emb_dict[s] = []

                enroll_emb_dict[s].append(cm_emb)

        for spk in enroll_emb_dict:
            enroll_emb_dict[spk] = Tensor(np.mean(enroll_emb_dict[spk], axis=0))

    return enroll_emb_dict

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
