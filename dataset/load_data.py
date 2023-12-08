import os.path

from torch.utils.data import DataLoader

from config import get_parser


def load_dataset(dataset_name, args, datatype):
    """ load dataset """
    # model name
    if args.hugging_face:
        model_name = args.model
    else:
        model_name = os.path.join(args.model_path, args.model)
    root_path = args.data_dir
    class_num = args.class_num
    max_length = args.max_length

    if dataset_name == 'COLD' or dataset_name == 'COLDataset':
        from dataset.COLD import COLDataset
        dataset = COLDataset(root_path, datatype, model_name, class_num, max_length)
    elif dataset_name == 'KOLD':
        from dataset.KOLD import KOLDataset
        dataset = KOLDataset(root_path, datatype, model_name, class_num, max_length)
    elif dataset_name == 'OLID':
        from dataset.OLID import OLIDataset
        dataset = OLIDataset(root_path, datatype, model_name, class_num, max_length)

    else:
        raise NotImplementedError

    return dataset


def get_dataloaders(args):
    """ get data loader """
    datasets = {'test': load_dataset(args.test_data, args, 'test'),
                'dev': load_dataset(args.dev_data, args, 'dev')}
    if args.transfer:
        datasets['source_train'] = load_dataset(args.source_data, args, 'train')
        datasets['target_train'] = load_dataset(args.target_data, args, 'train')
    else:
        datasets['train'] = load_dataset(args.train_data, args, 'train')

    dataloaders = {}

    for datatype, dataset in datasets.items():
        if 'train' in datatype:
            dataloaders[datatype] = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        else:
            dataloaders[datatype] = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    return dataloaders


def test_get_dataloaders():
    args = get_parser()
    print(args)
    dataloaders = get_dataloaders(args)
    print(dataloaders)


if __name__ == '__main__':
    test_get_dataloaders()
