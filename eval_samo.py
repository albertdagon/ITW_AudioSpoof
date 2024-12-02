import os
import argparse
import json
from pathlib import Path
from shutil import copy

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import set_seed, get_loader, get_model
from evaluation_utils import calculate_tDCF_EER, compute_eer, calculate_tDCF_EER
from data_utils import genSpoof_list, Dataset_ASV
from loss import SAMO

def eval_model(args, train_eval_model=None):
    # Parse configuration file
    with open(args.config, "r") as f_json:
        config = json.loads(f_json.read())
    model_config = config["model_config"]
    loss_config = config["loss"]
    optim_config = config["optim_config"]
    optim_config["epochs"] = config["num_epochs"]
    track = config["track"]

    if train_eval_model is not None:
        config["model_path"] = train_eval_model

    # For reproducibility
    set_seed(args.seed, config)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    output_dir = Path(args.output_dir)
    database_path = Path(config["database_path"])

    if track == "LA":
        eval_trial_path = (
            database_path / "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt"
        )
    elif track == "ITW":
        eval_trial_path = database_path / "in_the_wild.protocol.txt"

    # Create file logs folder for checkpoints
    model_logs = "LA_{}_ep{}_bs{}_train".format(
        os.path.splitext(os.path.basename(args.config))[0],
        config["num_epochs"],
        config["batch_size"],
    )
    model_logs = output_dir / model_logs
    model_save_path = model_logs / "weights"
    eval_score_path = model_logs / config["eval_output"]
    os.makedirs(model_save_path, exist_ok=True)
    copy(args.config, model_logs / "config.conf")

    # Define model
    model = get_model(model_config, device)

    # Define eval loaders
    if track == "LA":
        _, _, eval_data_loader, _, _, _, _ = get_loader(database_path, args.seed, config)
    elif track == "ITW":
        label_eval, file_eval, utt2spk_eval, tag_eval = genSpoof_list(
            dir_meta=eval_trial_path, enroll=False, is_train=False, is_itw=True
        )
        eval_set = Dataset_ASV(
            list_IDs=file_eval,
            labels=label_eval,
            utt2spk=utt2spk_eval,
            base_dir=database_path,
            tag_list=tag_eval,
            train=False,
            itw=True
        )
        eval_data_loader = DataLoader(eval_set,
                                    batch_size=config["batch_size"],
                                    shuffle=False,
                                    drop_last=False,
                                    pin_memory=True)

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

    # Enable eval mode and load loss func
    model.eval()
    samo = SAMO(loss_config["enc_dim"], m_real=loss_config["m_real"], m_fake=loss_config["m_fake"], alpha=loss_config["alpha"]).to(device)

    # Start evaluation
    with torch.no_grad():
        ip1_loader, utt_loader, idx_loader, score_loader, spk_loader, tag_loader = [], [], [], [], [], []

        # Store list of speakers in dataset
        if track == "LA":
            spklist = ['LA_00' + str(spk_id) for spk_id in range(79, 99)]
        elif track == "ITW":
            spklist = []
            with open (eval_trial_path, "r") as f:
                lines = f.readlines()
                for line in lines:
                    _, spk, _ = line.strip().split(" ")
                    spklist.append(spk)
            spklist = list(set(spklist))

        tmp_center = torch.eye(loss_config["enc_dim"])[:20]
        eval_enroll = dict(zip(spklist, tmp_center))
        samo.center = torch.stack(list(eval_enroll.values()))

        for i, (feat, labels, spk, utt, tag) in enumerate(tqdm(eval_data_loader)):
            feat = feat.to(device)
            labels = labels.to(device)
            feats, _ = model(feat)

            _, score = samo.inference(feats, labels, spk, eval_enroll)

            ip1_loader.append(feats)
            idx_loader.append(labels)
            score_loader.append(score)
            utt_loader.extend(utt)
            spk_loader.extend(spk)
            tag_loader.extend(tag)

        scores = torch.cat(score_loader, 0).data.cpu().numpy()
        labels = torch.cat(idx_loader, 0).data.cpu().numpy()
        eer = compute_eer(scores[labels == 0], scores[labels == 1])[0]
    
    # Save scores
    with open(eval_score_path, "w") as fh: 
        for utt, tag, score, label, spk in zip(utt_loader, tag_loader, scores, labels, spk_loader):
            fh.write(f"{utt} {tag} {label} {score} {spk}\n")
    print(f"Scores saved to {eval_score_path}")

    # Calc EER based on dataset (do t-dcf for LA only)
    print(f"Test EER: {eer*100}%")
    if track == "LA":
        calculate_tDCF_EER(cm_scores_file=eval_score_path,
            asv_score_file=database_path / config["asv_score_path"],
            output_file=model_logs / "t-DCF_EER.txt")
    elif track == "ITW":
        eval_itw(eval_score_path)

def eval_itw(score_file):
    # Load CM scores
    cm_data = np.genfromtxt(score_file, dtype=str)
    cm_keys = cm_data[:, 2].astype(int)
    cm_scores = cm_data[:, 3].astype(float)

    # Extract bona fide (real human) and spoof scores from the CM scores
    bona_cm = cm_scores[cm_keys == 0]
    spoof_cm = cm_scores[cm_keys == 1]
    eer_cm = compute_eer(bona_cm, spoof_cm)[0]
    print(f"Final EER: {eer_cm*100}%")

def main(args):
    eval_model(args)

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

    # Add the --gpu argument here                                                                                                                                                                               
    parser.add_argument(
        "--gpu", type=int, default=0, help="GPU id to use (default: 0)"
    )

    
    args = parser.parse_args()
    main(args)
