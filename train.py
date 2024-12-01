import os
import argparse
import json
from pathlib import Path
from shutil import copy

import torch
from torch import nn

from torch.utils.tensorboard import SummaryWriter
from torchcontrib.optim import SWA

from utils import create_optimizer, set_seed, str_to_bool, get_model, get_loader
from evaluation_utils import calculate_tDCF_EER, produce_evaluation_file


def main(args):
    # Parse configuration file
    with open(args.config, "r") as f_json:
        config = json.loads(f_json.read())
    model_config = config["model_config"]
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
    database_path = Path(config["database_path"])
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
    train_loader, dev_loader, eval_loader = get_loader(database_path, args.seed, config)

    # Define optimizer and scheduler
    optim_config["steps_per_epoch"] = len(train_loader)
    optimizer, scheduler = create_optimizer(model.parameters(), optim_config)
    optimizer_swa = SWA(optimizer)

    best_dev_eer = 1.0
    best_eval_eer = 100.0
    best_dev_tdcf = 0.05
    best_eval_tdcf = 1.0
    n_swa_update = 0  # number of snapshots of model to use in SWA
    f_log = open(model_logs / "metric_log.txt", "a")
    f_log.write("=" * 5 + "\n")

    # make directory for metric logging
    metric_path = model_logs / "metrics"
    os.makedirs(metric_path, exist_ok=True)

    # Train the model
    for epoch in range(config["num_epochs"]):
        print("Start training epoch{:03d}".format(epoch))
        running_loss = train_epoch(
            train_loader, model, optimizer, device, scheduler, config
        )
        produce_evaluation_file(
            dev_loader, model, device, metric_path / "dev_score.txt", dev_trial_path
        )
        dev_eer, dev_tdcf = calculate_tDCF_EER(
            cm_scores_file=metric_path / "dev_score.txt",
            asv_score_file=database_path / config["asv_score_path"],
            output_file=metric_path / "dev_t-DCF_EER_{}epo.txt".format(epoch),
            printout=False,
        )
        print(
            "DONE.\nLoss:{:.5f}, dev_eer: {:.3f}, dev_tdcf:{:.5f}".format(
                running_loss, dev_eer, dev_tdcf
            )
        )

        writer.add_scalar("loss", running_loss, epoch)
        writer.add_scalar("dev_eer", dev_eer, epoch)
        writer.add_scalar("dev_tdcf", dev_tdcf, epoch)

        best_dev_tdcf = min(dev_tdcf, best_dev_tdcf)
        if best_dev_eer >= dev_eer:
            print("best model find at epoch", epoch)
            best_dev_eer = dev_eer
            torch.save(
                model.state_dict(),
                model_save_path / "epoch_{}_{:03.3f}.pth".format(epoch, dev_eer),
            )

            # do evaluation whenever best model is renewed
            if str_to_bool(config["eval_all_best"]):
                produce_evaluation_file(
                    eval_loader, model, device, eval_score_path, eval_trial_path
                )
                eval_eer, eval_tdcf = calculate_tDCF_EER(
                    cm_scores_file=eval_score_path,
                    asv_score_file=database_path / config["asv_score_path"],
                    output_file=metric_path / "t-DCF_EER_{:03d}epo.txt".format(epoch),
                )

                log_text = "epoch{:03d}, ".format(epoch)
                if eval_eer < best_eval_eer:
                    log_text += "best eer, {:.4f}%".format(eval_eer)
                    best_eval_eer = eval_eer
                if eval_tdcf < best_eval_tdcf:
                    log_text += "best tdcf, {:.4f}".format(eval_tdcf)
                    best_eval_tdcf = eval_tdcf
                    torch.save(model.state_dict(), model_save_path / "best.pth")
                if len(log_text) > 0:
                    print(log_text)
                    f_log.write(log_text + "\n")

            print("Saving epoch {} for swa".format(epoch))
            optimizer_swa.update_swa()
            n_swa_update += 1
        writer.add_scalar("best_dev_eer", best_dev_eer, epoch)
        writer.add_scalar("best_dev_tdcf", best_dev_tdcf, epoch)

    print("Start final evaluation")
    epoch += 1
    if n_swa_update > 0:
        optimizer_swa.swap_swa_sgd()
        optimizer_swa.bn_update(train_loader, model, device=device)
    produce_evaluation_file(
        eval_loader, model, device, eval_score_path, eval_trial_path
    )
    eval_eer, eval_tdcf = calculate_tDCF_EER(
        cm_scores_file=eval_score_path,
        asv_score_file=database_path / config["asv_score_path"],
        output_file=model_logs / "t-DCF_EER.txt",
    )
    f_log = open(model_logs / "metric_log.txt", "a")
    f_log.write("=" * 5 + "\n")
    f_log.write("EER: {:.3f}, min t-DCF: {:.5f}".format(eval_eer, eval_tdcf))
    f_log.close()

    torch.save(model.state_dict(), model_save_path / "swa.pth")

    if eval_eer <= best_eval_eer:
        best_eval_eer = eval_eer
    if eval_tdcf <= best_eval_tdcf:
        best_eval_tdcf = eval_tdcf
        torch.save(model.state_dict(), model_save_path / "best.pth")
    print(
        "Exp FIN. EER: {:.3f}, min t-DCF: {:.5f}".format(best_eval_eer, best_eval_tdcf)
    )


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
        "--gpu", type=int, default=0, help="GPU id to use (default: 0)"
    )
    parser.add_argument(
        "--seed", type=int, default=40, help="random seed (default: 40)"
    )

    args = parser.parse_args()
    main(args)
