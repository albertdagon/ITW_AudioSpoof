import os
import argparse
import json
from pathlib import Path
from shutil import copy
from collections import defaultdict

import numpy as np
import torch
from torch import nn, Tensor

from torch.utils.tensorboard import SummaryWriter
from torchcontrib.optim import SWA

from utils import create_optimizer, set_seed, str_to_bool, get_model, get_loader
from evaluation_utils import compute_eer
from eval_samo import eval_model
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
    model_data_logs = model_logs / "logs"
    # eval_score_path = model_logs / config["eval_output"]
    os.makedirs(model_save_path, exist_ok=True)
    os.makedirs(model_data_logs, exist_ok=True)
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
    monitor_loss = "samo"
    samo = SAMO(loss_config["enc_dim"], m_real=loss_config["m_real"], m_fake=loss_config["m_fake"], alpha=loss_config["alpha"],
                    num_centers=loss_config["num_centers"]).to(device)

    # early_stop setup
    early_stop_cnt = 0
    prev_loss = 1e8
    best_epoch = 0

    # number of snapshots of model to use in SWA
    n_swa_update = 0

    # Start training
    for epoch in tqdm(range(config["num_epochs"])):
        model.train()

        ip1_loader, idx_loader = [], []
        trainloss_dict = defaultdict(list)
        devloss_dict = defaultdict(list)

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
            feats, _ = model(feat)

            # Calc loss
            samoloss, _ = samo(feats, labels, spk=spk, enroll=train_enroll, attractor=1)
            feat_loss = samoloss

            # Backward steps
            trainloss_dict[monitor_loss].append(feat_loss.item())
            optimizer.zero_grad()
            feat_loss.backward()
            optimizer.step()
            
            # LR adjustment
            if config["optim_config"]["scheduler"] in ["cosine", "keras_decay"]:
                scheduler.step()
            
            # Record results
            ip1_loader.append(feats)
            idx_loader.append((labels))

            # Write loss to file
            with open(os.path.join(model_data_logs, "train_loss.log"), "a") as log:
                log.write(str(epoch) + "\t" + str(i) + "\t" +
                          str(trainloss_dict[monitor_loss][-1]) + "\n")
            
            # Save center
            with open(os.path.join(model_data_logs, "train_enroll.log"), "a") as log:
                log.write(str(epoch) + "\t" + str(samo.center.detach().numpy()) + "\n")
            
            
        # Validate the model
        print("Begin validation...\n")
        model.eval()
        with torch.no_grad():
            ip1_loader, idx_loader, score_loader = [], [], []

            print("Updating embeddings...\n")
            dev_enroll = update_embeds(device, model, dev_enroll_loader)
            samo.center = torch.stack(list(dev_enroll.values()))

            print("Validating...\n")
            for i, (feat, labels, spk, utt, tag) in enumerate(tqdm(dev_data_loader)):
                feat = feat.to(device)
                labels = labels.to(device)
                feats, _ = model(feat)
                samoloss, score = samo.inference(feats, labels, spk, dev_enroll, attractor=1)
                devloss_dict[monitor_loss].append(samoloss.item())
                ip1_loader.append(feats)
                idx_loader.append(labels)
                score_loader.append(score)
                
            scores = torch.cat(score_loader, 0).data.cpu().numpy()
            labels = torch.cat(idx_loader, 0).data.cpu().numpy()
            eer = compute_eer(scores[labels == 0], scores[labels == 1])[0]

            with open(os.path.join(model_data_logs, "dev_loss.log"), "a") as log:
                log.write(str(epoch) + "\t" +
                            str(np.nanmean(devloss_dict[monitor_loss])) + "\t" +
                            str(eer) + "\n")
            print("Val EER: {}".format(eer))

            with open(os.path.join(model_data_logs, "dev_enroll{}.log".format(epoch)), "a") as log:
                log.write(str(epoch) + "\t" + str(samo.center.detach().numpy()) + "\n")

        # save best model by lowest val loss
        valLoss = np.nanmean(devloss_dict[monitor_loss])
        if valLoss < prev_loss:
            torch.save(model.state_dict(), model_save_path / "best.pth")
            loss_model = samo
            torch.save(loss_model.state_dict(), model_save_path / "loss_model.pth")

            prev_loss = valLoss
            early_stop_cnt = 0
            best_epoch = epoch
            optimizer_swa.update_swa()
            n_swa_update += 1
        else:
            early_stop_cnt += 1

        if early_stop_cnt == 100:
            break

    eval_model(args, model_save_path / "best.pth")
    print("Saving best model in epoch {}\n".format(best_epoch))


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
