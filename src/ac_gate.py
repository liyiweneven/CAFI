import torch
import torch.multiprocessing as mp
import os
import sys


from utils.manager import Manager
from models.AC_gate import AC_gate
from torch.nn.parallel import DistributedDataParallel as DDP


def main(rank, manager):
    """ train/dev/test the model (in distributed)

    Args:
        rank: current process id
        world_size: total gpus
    """
    manager.setup(rank)
    loaders = manager.prepare()

    model = AC_gate(manager).to(manager.device)

    if manager.mode == 'train':
        if manager.world_size > 1:
            model = DDP(model, device_ids=[manager.device], output_device=manager.device,find_unused_parameters=True)

        manager.train(model, loaders)

    elif manager.mode == 'dev':
        manager.load(model)
        model.dev(manager, loaders, log=True)

    elif manager.mode == 'test':
        manager.load(model)
        model.test(manager, loaders)


if __name__ == "__main__":
    config = {
        "mode": 'train',
        "epochs": 2,
        "batch_size": 6,
        "batch_size_eval": 10,
        "enable_fields": ["title"],
        "hidden_dim": 150,
        "learning_rate": 1e-5,
        "validate_step": "0.5e",#Bert模型比较大，所以跑一半验证一下
        "seed": 3406,
    }
    manager = Manager(config)

    # essential to set this to False to speed up dilated cnn
    torch.backends.cudnn.deterministic = False

    if manager.world_size > 1:
        mp.spawn(
            main,
            args=(manager,),
            nprocs=manager.world_size,
            join=True
        )
    else:
        main(manager.device, manager)