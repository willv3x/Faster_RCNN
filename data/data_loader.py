from torch.utils.data import DataLoader


def collate_fn(batch):
    return tuple(zip(*batch))


def build_data_loader(dataset, batch_size, shuffle, num_workers, drop_last):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, drop_last=drop_last, collate_fn=collate_fn)
