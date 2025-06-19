import torch.utils.data

from datasets.base_dataset import CreateDataset
from datasets.base_dataset import data_sampler

def get_data_generator(loader):
    while True:
        for data in loader:
            yield data

def CreateDataLoader(opt):
    train_dataset, test_dataset = CreateDataset(opt)
    print(f"Creating DataLoader with num_workers = {opt.nThreads}")

    loader_kwargs = {
        "batch_size": opt.batch_size,
        "sampler": data_sampler(train_dataset, shuffle=True, distributed=opt.distributed),
        "drop_last": True,
        "num_workers": opt.nThreads,
        "pin_memory": False,           # try False to avoid any extra C allocations
        "persistent_workers": False,   # keep this off until it’s stable
    }
    train_dl =torch.utils.data.DataLoader(train_dataset, **loader_kwargs)

    test_kwargs = loader_kwargs.copy()

    test_kwargs.update({
        "sampler": data_sampler(test_dataset, shuffle=False, distributed=opt.distributed),
        "drop_last": False,
    })

    test_dl = torch.utils.data.DataLoader(test_dataset, **test_kwargs)
    # train_dl = torch.utils.data.DataLoader(
    #         train_dataset,
    #         batch_size=opt.batch_size,
    #         sampler=data_sampler(train_dataset, shuffle=True, distributed=opt.distributed),
    #         drop_last=True,
           
           
    #         )

    # test_dl = torch.utils.data.DataLoader(
    #         test_dataset,
    #         batch_size=opt.batch_size,
    #         sampler=data_sampler(test_dataset, shuffle=False, distributed=opt.distributed),
    #         drop_last=False, 
            
           
    #         )
    eval_kwargs = test_kwargs.copy()

    eval_kwargs["batch_size"] = max(opt.batch_size // 2, 1)
    test_dl_for_eval = torch.utils.data.DataLoader(test_dataset, **eval_kwargs)
    # test_dl_for_eval = torch.utils.data.DataLoader(
    #         test_dataset,
    #         batch_size=max(int(opt.batch_size // 2), 1),
    #         sampler=data_sampler(test_dataset, shuffle=False, distributed=opt.distributed),
    #         drop_last=False,
           
           
    #     )

    return train_dl, test_dl, test_dl_for_eval
