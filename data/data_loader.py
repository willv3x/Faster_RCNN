from torch.utils.data import DataLoader


def _collate_fn(batch):
    return tuple(zip(*batch))


def data_loader(dataset, batch_size, shuffle, num_workers, drop_last):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, drop_last=drop_last, collate_fn=_collate_fn)


if __name__ == '__main__':
    print('hi')

