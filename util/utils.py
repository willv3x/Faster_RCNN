import torch


def save_model(model, model_name):
    torch.save(model.state_dict(), f'{model_name}.pt')
