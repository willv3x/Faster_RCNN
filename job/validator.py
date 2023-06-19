import torch
from torchmetrics.detection import MeanAveragePrecision
from tqdm import tqdm

from util.utils import save_model


class Validator:
    def __init__(self):
        self._best_epoch_validation_metric = None

    def validate_one_epoch(self, model, device, validation_loader, epoch):
        model.eval()

        batches_map = []
        all_targets = []
        all_predictions = []

        progress_bar = tqdm(validation_loader,
                            total=len(validation_loader), unit='batch', desc=f"Epoch: {epoch}", colour='red')

        for batch, (images, targets) in enumerate(validation_loader):
            images = list(image.to(device) for image in images)

            with torch.no_grad():
                predictions = model(images, targets)

            batch_targets = []
            batch_predictions = []
            for prediction, target in zip(predictions, targets):
                target_dict = dict()
                prediction_dict = dict()

                target_dict['boxes'] = target['boxes'].detach().cpu()
                target_dict['labels'] = target['labels'].detach().cpu()

                prediction_dict['boxes'] = prediction['boxes'].detach().cpu()
                prediction_dict['scores'] = prediction['scores'].detach().cpu()
                prediction_dict['labels'] = prediction['labels'].detach().cpu()

                all_targets.append(target_dict)
                all_predictions.append(prediction_dict)

                batch_targets.append(target_dict)
                batch_predictions.append(prediction_dict)

            batch_map = MeanAveragePrecision()
            batch_map.update(batch_predictions, batch_targets)
            batch_map = batch_map.compute()

            batch_map_50_95 = batch_map['map'].item()

            batches_map.append(batch_map_50_95)

            progress_bar.set_description(
                desc=f"Validating epoch {epoch}, Batch {batch + 1} mAP_0.5:0.95 {batch_map_50_95:.4f}")
            progress_bar.update(1)

        epoch_map = MeanAveragePrecision()
        epoch_map.update(all_predictions, all_targets)
        epoch_map = epoch_map.compute()

        epoch_map_50_95 = epoch_map['map'].item()
        epoch_map_50 = epoch_map['map_50'].item()

        validation_data = {
            "metrics/mAP_0.5": epoch_map_50,
            "metrics/mAP_0.5:0.95": epoch_map_50_95,
            "metrics/batches_mAP_0.5:0.95": batches_map
        }

        progress_bar.close()

        print(f"Validating epoch {epoch} finished. Epoch validation mAP_0.5:0.95: {epoch_map_50_95:.4f}")

        if self._best_epoch_validation_metric is None \
                or self._best_epoch_validation_metric < validation_data['metrics/mAP_0.5:0.95']:
            self._best_epoch_validation_metric = validation_data['metrics/mAP_0.5:0.95']
            print(f"Saving model with best epoch validation mAP_0.5:0.95: {validation_data['metrics/mAP_0.5:0.95']} "
                  f"(at epoch {epoch})")
            save_model(model, 'best_validation_model')

        return validation_data


if __name__ == '__main__':
    print('hi')
