import os
import argparse
import json
from pathlib import Path
from shutil import copy
from collections import defaultdict

import torch
from torch import nn

from torch.utils.tensorboard import SummaryWriter
from torchcontrib.optim import SWA

from utils import create_optimizer, set_seed, str_to_bool, get_model, get_loader
from evaluation_utils import calculate_tDCF_EER, produce_evaluation_file
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
    database_path = Path("./datasets/LA/")

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

    # load datasets
    train_data_loader, dev_data_loader, eval_data_loader, \
    train_bona_loader, dev_enroll_loader, eval_enroll_loader, num_centers = get_loader(database_path, args.seed, config)
    
    # Define optimizer and scheduler
    optim_config["steps_per_epoch"] = len(train_data_loader)
    optimizer, scheduler = create_optimizer(model.parameters(), optim_config)
    optimizer_swa = SWA(optimizer)

    # Loss
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([0.9, 0.1])).to(args.device)
    monitor_loss = "samo"
    samo = SAMO(loss_config["enc_dim"], m_real=loss_config["m_real"], m_fake=loss_config["m_fake"], alpha=loss_config["alpha"],
                    num_centers=loss_config["num_centers"], initialize_centers=loss_config["initialize_centers"]).to(device)
    ocsoftmax_optimizer = None

    # early_stop setup
    early_stop_cnt = 0
    prev_loss = 1e8
    best_epoch = 0

    # number of snapshots of model to use in SWA
    n_swa_update = 0

    # Styart training
    for epoch in tqdm(range(config["num_epochs"])):
        model.train()

        ip1_loader, idx_loader, spk_loader, utt_loader = [], [], [], []
        trainloss_dict = defaultdict(list)
        devloss_dict = defaultdict(list)
        testloss_dict = defaultdict(list)
        print('\nEpoch: %d ' % (epoch + 1))

        if epoch == 0:
            spklist = ['LA_00' + str(spk_id) for spk_id in range(79, 99)]
            tmp_center = torch.eye(loss_config["enc_dim"])[:num_centers[0]]
            train_enroll = dict(zip(spklist, tmp_center))
        elif epoch % 3 == 0:
            train_enroll = update_embeds(device, model, train_bona_loader)
            
        # Pass centers to loss
        samo.center = torch.stack(list(train_enroll.values()))

        for i, (feat, labels, spk, utt, tag) in enumerate(tqdm(train_data_loader)):
            feat = feat.to(device)
            labels = labels.to(device)

            # forward
            feats, feat_outputs = model(feat)
            samoloss, _ = samo(feats, labels, spk=spk, enroll=train_enroll, attractor=1)
            feat_loss = samoloss
            trainloss_dict[monitor_loss].append(feat_loss.item())
            optimizer.zero_grad()
            feat_loss.backward()
            optimizer.step()
            
            if config["optim_config"]["scheduler"] in ["cosine", "keras_decay"]:
                scheduler.step()
            
            # record
            ip1_loader.append(feats)
            idx_loader.append((labels))

            # Log loss
            with open(os.path.join("exp_result", "train_loss.log"), "a") as log:
                log.write(str(epoch) + "\t" + str(i) + "\t" +
                          str(trainloss_dict[monitor_loss][-1]) + "\n")
            
            # Save center
            with open(os.path.join("exp_result", "train_enroll.log"), "a") as log:
                log.write(str(epoch) + "\t" + str(samo.center.detach().numpy()) + "\n")
            
            # Val the model
            model.eval()
            with torch.no_grad():
                ip1_loader, idx_loader, spk_loader, score_loader = [], [], [], []

                dev_enroll = update_embeds(device, model, dev_enroll_loader)
                samo.center = torch.stack(list(dev_enroll.values()))

                for i, (feat, labels, spk, utt, tag) in enumerate(tqdm(dev_data_loader)):
                    feat = feat.to(args.device)
                    labels = labels.to(args.device)
                    feats, feat_outputs = model(feat)
                    samoloss, score = samo(feats, labels, spk=spk, enroll=dev_enroll, attractor=1)


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
            enroll_emb_dict[spk] = torch.mean(enroll_emb_dict[spk], axis=0)

    return enroll_emb_dict




def train_epoch(trn_loader, model, optim, device, scheduler, config):
    """Train the model for one epoch"""
    running_loss = 0
    num_total = 0.0
    ii = 0
    model.train()

    # set objective (Loss) functions
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    for batch_x, batch_y in trn_loader:
        batch_size = batch_x.size(0)
        num_total += batch_size
        ii += 1
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        _, batch_out = model(batch_x, Freq_aug=str_to_bool(config["freq_aug"]))
        batch_loss = criterion(batch_out, batch_y)
        running_loss += batch_loss.item() * batch_size
        optim.zero_grad()
        batch_loss.backward()
        optim.step()

        if config["optim_config"]["scheduler"] in ["cosine", "keras_decay"]:
            scheduler.step()
        elif scheduler is None:
            pass
        else:
            raise ValueError("scheduler error, got:{}".format(scheduler))

    running_loss /= num_total
    return running_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training for audio spoofing detection system")
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

    args = parser.parse_args()
    main(args)
