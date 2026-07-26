import torch.utils.data

from datasets.base_dataset import CreateDataset
from datasets.base_dataset import data_sampler

def get_data_generator(loader):
    while True:
        for data in loader:
            yield data

def CreateDataLoader(opt):
    train_dataset, val_dataset, test_dataset = CreateDataset(opt)
    print(f"Creating DataLoader with num_workers = {opt.nThreads}")
    train_dl = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=opt.batch_size,
            sampler=data_sampler(train_dataset, shuffle=True, distributed=opt.distributed),
            drop_last=True,
            )

    val_dl = None
    if val_dataset is not None:
        val_dl = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=opt.batch_size,
                sampler=data_sampler(val_dataset, shuffle=False, distributed=opt.distributed),
                drop_last=False,
                )

    test_dl = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=opt.batch_size,
            sampler=data_sampler(test_dataset, shuffle=False, distributed=opt.distributed),
            drop_last=False,
            )

    test_dl_for_eval = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=max(int(opt.batch_size // 2), 1),
            sampler=data_sampler(test_dataset, shuffle=False, distributed=opt.distributed),
            drop_last=False,
        )

    return train_dl, val_dl, test_dl, test_dl_for_eval
