from torch.utils.data import DataLoader


def build_data_loader(dataset, batch_size, shuffle, num_workers, drop_last):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last,
                      collate_fn=lambda batch: tuple(zip(*batch)))
