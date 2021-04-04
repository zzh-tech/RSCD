from importlib import import_module


class Data:
    def __init__(self, para):
        dataset = para.dataset
        module = import_module('data.' + dataset)
        self.dataloader_train = module.Dataloader(para, ds_type='train')
        self.dataloader_valid = module.Dataloader(para, ds_type='valid')
