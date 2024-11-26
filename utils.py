"""
Utilization functions
"""

import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from data_utils import genSpoof_list, Dataset_ASV
from importlib import import_module


def str_to_bool(val):
    """Convert a string representation of truth to true (1) or false (0).
    Copied from the python implementation distutils.utils.strtobool

    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
    are 'n', 'no', 'f', 'false', 'off', and '0'.  Raises ValueError if
    'val' is anything else.
    >>> str_to_bool('YES')
    1
    >>> str_to_bool('FALSE')
    0
    """
    val = val.lower()
    if val in ('y', 'yes', 't', 'true', 'on', '1'):
        return True
    if val in ('n', 'no', 'f', 'false', 'off', '0'):
        return False
    raise ValueError('invalid truth value {}'.format(val))


def cosine_annealing(step, total_steps, lr_max, lr_min):
    """Cosine Annealing for learning rate decay scheduler"""
    return lr_min + (lr_max -
                     lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))


def keras_decay(step, decay=0.0001):
    """Learning rate decay in Keras-style"""
    return 1. / (1. + decay * step)


class SGDRScheduler(torch.optim.lr_scheduler._LRScheduler):
    """SGD with restarts scheduler"""
    def __init__(self, optimizer, T0, T_mul, eta_min, last_epoch=-1):
        self.Ti = T0
        self.T_mul = T_mul
        self.eta_min = eta_min

        self.last_restart = 0

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        T_cur = self.last_epoch - self.last_restart
        if T_cur >= self.Ti:
            self.last_restart = self.last_epoch
            self.Ti = self.Ti * self.T_mul
            T_cur = 0

        return [
            self.eta_min + (base_lr - self.eta_min) *
            (1 + np.cos(np.pi * T_cur / self.Ti)) / 2
            for base_lr in self.base_lrs
        ]


def _get_optimizer(model_parameters, optim_config):
    """Defines optimizer according to the given config"""
    optimizer_name = optim_config['optimizer']

    if optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(model_parameters,
                                    lr=optim_config['base_lr'],
                                    momentum=optim_config['momentum'],
                                    weight_decay=optim_config['weight_decay'],
                                    nesterov=optim_config['nesterov'])
    elif optimizer_name == 'adam':
        optimizer = torch.optim.Adam(model_parameters,
                                     lr=optim_config['base_lr'],
                                     betas=optim_config['betas'],
                                     weight_decay=optim_config['weight_decay'],
                                     amsgrad=str_to_bool(
                                         optim_config['amsgrad']))
    else:
        print('Un-known optimizer', optimizer_name)
        sys.exit()

    return optimizer


def _get_scheduler(optimizer, optim_config):
    """
    Defines learning rate scheduler according to the given config
    """
    if optim_config['scheduler'] == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=optim_config['milestones'],
            gamma=optim_config['lr_decay'])

    elif optim_config['scheduler'] == 'sgdr':
        scheduler = SGDRScheduler(optimizer, optim_config['T0'],
                                  optim_config['Tmult'],
                                  optim_config['lr_min'])

    elif optim_config['scheduler'] == 'cosine':
        total_steps = optim_config['epochs'] * \
            optim_config['steps_per_epoch']

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: cosine_annealing(
                step,
                total_steps,
                1,  # since lr_lambda computes multiplicative factor
                optim_config['lr_min'] / optim_config['base_lr']))

    elif optim_config['scheduler'] == 'keras_decay':
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda step: keras_decay(step))
    else:
        scheduler = None
    return scheduler


def create_optimizer(model_parameters, optim_config):
    """Defines an optimizer and a scheduler"""
    optimizer = _get_optimizer(model_parameters, optim_config)
    scheduler = _get_scheduler(optimizer, optim_config)
    return optimizer, scheduler


def seed_worker(worker_id):
    """
    Used in generating seed for the worker of torch.utils.data.Dataloader
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def set_seed(seed, config = None):
    """ 
    set initial seed for reproduction
    """
    if config is None:
        raise ValueError("config should not be None")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = str_to_bool(config["cudnn_deterministic_toggle"])
        torch.backends.cudnn.benchmark = str_to_bool(config["cudnn_benchmark_toggle"])

def get_model(model_config, device):
    """Define DNN model architecture"""
    module = import_module("models.{}".format(model_config["architecture"]))
    _model = getattr(module, "Model")
    model = _model(model_config).to(device)
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    print("no. model params:{}".format(nb_params))

    return model

def get_loader(database_path, seed, config):
    """Return list of PyTorch DataLoaders for train / developement"""

    trn_database_path = database_path / "ASVspoof2019_LA_train/"
    dev_database_path = database_path / "ASVspoof2019_LA_dev/"
    eval_database_path = database_path / "ASVspoof2019_LA_eval/"

    trn_list_path = (
        database_path / "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt"
    )
    dev_trial_path = (
        database_path / "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt"
    )
    eval_trial_path = (
        database_path / "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt"
    )

    dev_enroll_path = [
        database_path
        / "ASVspoof2019_LA_asv_protocols/ASVspoof2019.LA.asv.dev.female.trn.txt",
        database_path
        / "ASVspoof2019_LA_asv_protocols/ASVspoof2019.LA.asv.dev.male.trn.txt",
    ]
    eval_enroll_path = [
        database_path
        / "ASVspoof2019_LA_asv_protocols/ASVspoof2019.LA.asv.eval.female.trn.txt",
        database_path
        / "ASVspoof2019_LA_asv_protocols/ASVspoof2019.LA.asv.eval.male.trn.txt",
    ]

    # Read all training data
    label_trn, file_train, utt2spk_train, tag_train = genSpoof_list(
        dir_meta=trn_list_path, enroll=False, is_train=True
    )
    trn_centers = len(set(utt2spk_train.values()))
    print("no. training files:", len(file_train))
    print("no. training speakers:", trn_centers)

    train_set = Dataset_ASV(
        list_IDs=file_train,
        labels=label_trn,
        utt2spk=utt2spk_train,
        base_dir=trn_database_path,
        tag_list=tag_train,
        train=True,
    )
    gen = torch.Generator()
    gen.manual_seed(seed)
    trn_loader = DataLoader(
        train_set,
        batch_size=config["batch_size"],
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=gen,
    )

    # Read bona-fide-only training data
    num_bonafide_train = 2580
    train_set_fix = Dataset_ASV(
        list_IDs=file_train,
        labels=label_trn,
        utt2spk=utt2spk_train,
        base_dir=trn_database_path,
        tag_list=tag_train,
        train=False,
    )
    trn_bona_set = Subset(train_set_fix, range(num_bonafide_train))
    trn_bona = DataLoader(
        trn_bona_set,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        drop_last=False,
        pin_memory=True,
    )

    # Read dev enrollment data
    label_dev_enroll, file_dev_enroll, utt2spk_dev_enroll, tag_dev_enroll = (
        genSpoof_list(dir_meta=dev_enroll_path, enroll=True, is_train=False)
    )
    dev_enroll_spk = set(utt2spk_dev_enroll.values())
    dev_centers = len(dev_enroll_spk)
    print(f"no. validation enrollment files: {len(file_dev_enroll)}")
    print(f"no. validation enrollment speakers: {dev_centers}")

    dev_set_enroll = Dataset_ASV(
        list_IDs=file_dev_enroll,
        labels=label_dev_enroll,
        utt2spk=utt2spk_dev_enroll,
        base_dir=dev_database_path,
        tag_list=tag_dev_enroll,
        train=False,
    )
    dev_enroll = DataLoader(
        dev_set_enroll,
        batch_size=config["batch_size"],
        shuffle=False,
        drop_last=False,
        pin_memory=True,
    )

    # Read target-only dev data
    label_dev, file_dev, utt2spk_dev, tag_dev = genSpoof_list(
        dir_meta=dev_trial_path,
        enroll=False,
        is_train=False,
        target_only=config["loss"]["target_only"],
        enroll_spk=dev_enroll_spk,
    )
    print(f"no. validation files: {len(file_dev)}")

    dev_set = Dataset_ASV(
        list_IDs=file_dev,
        labels=label_dev,
        utt2spk=utt2spk_dev,
        base_dir=dev_database_path,
        tag_list=tag_dev,
        train=False,
    )
    dev_loader = DataLoader(
        dev_set,
        batch_size=config["batch_size"],
        shuffle=False,
        drop_last=False,
        pin_memory=True,
    )

    # Read eval enrollment data
    label_eval_enroll, file_eval_enroll, utt2spk_eval_enroll, tag_eval_enroll = (
        genSpoof_list(dir_meta=eval_enroll_path, enroll=True, is_train=False)
    )
    eval_enroll_spk = set(utt2spk_eval_enroll.values())
    eval_centers = len(eval_enroll_spk)
    print(f"no. eval enrollment files: {len(file_eval_enroll)}")
    print(f"no. eval enrollment speakers: {eval_centers}")

    eval_set_enroll = Dataset_ASV(
        list_IDs=file_eval_enroll,
        labels=label_eval_enroll,
        utt2spk=utt2spk_eval_enroll,
        base_dir=eval_database_path,
        tag_list=tag_eval_enroll,
        train=False,
    )
    eval_enroll = DataLoader(
        eval_set_enroll,
        batch_size=config["batch_size"],
        shuffle=False,
        drop_last=False,
        pin_memory=True,
    )

    # Read eval target-only data
    label_eval, file_eval, utt2spk_eval, tag_eval = genSpoof_list(
        dir_meta=eval_trial_path,
        enroll=False,
        is_train=False,
        target_only=config["loss"]["target_only"],
        enroll_spk=eval_enroll_spk,
    )

    # Define eval loader
    print(f"no. eval files: {len(file_eval)}")
    eval_set = Dataset_ASV(
        list_IDs=file_eval,
        labels=label_eval,
        utt2spk=utt2spk_eval,
        base_dir=eval_database_path,
        tag_list=tag_eval,
        train=False,
    )
    eval_loader = DataLoader(
        eval_set,
        batch_size=config["batch_size"],
        shuffle=False,
        drop_last=False,
        pin_memory=True,
    )

    num_centers = [trn_centers, dev_centers, eval_centers]

    return (
        trn_loader,
        dev_loader,
        eval_loader,
        trn_bona,
        dev_enroll,
        eval_enroll,
        num_centers,
    )