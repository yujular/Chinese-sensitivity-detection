def load_dataset(dataset_name, args, datatype='train'):
    """ load dataset """
    if dataset_name == 'COLD':
        from dataset.COLD import COLDataset
        dataset = COLDataset(args, datatype)
    elif dataset_name == 'OLID':
        from dataset.OLID import OLIDataset
        dataset = OLIDataset(args, datatype)
    else:
        raise NotImplementedError

    return dataset
