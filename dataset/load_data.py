from torch.utils.data import DataLoader


def load_dataset(dataset_name, args, datatype='train'):
    """ load dataset """
    if dataset_name == 'COLD' or dataset_name == 'COLDataset':
        from dataset.COLD import COLDataset
        dataset = COLDataset(args, datatype)
    elif dataset_name == 'OLID':
        from dataset.OLID import OLIDataset
        dataset = OLIDataset(args, datatype)
    elif dataset_name == 'KOLD':
        from dataset.KOLD import KOLDataset
        dataset = KOLDataset(args, datatype)
    else:
        raise NotImplementedError

    return dataset


def get_dataloader(dataset_name, args, datatypes):
    """ get data loader """
    dataloader = {}
    for datatype in datatypes:
        dataset = load_dataset(dataset_name, args, datatype)
        if datatype == 'train':
            dataloader[datatype] = DataLoader(dataset, batch_size=args.train['batch_size'], shuffle=True)
        else:
            dataloader[datatype] = DataLoader(dataset, batch_size=args.train['batch_size'], shuffle=False)
    return dataloader
