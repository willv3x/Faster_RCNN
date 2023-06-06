import torch
import wandb
from torchmetrics.detection import MeanAveragePrecision
from tqdm import tqdm


def _save_model(model, epoch_validation_metric, epoch):
    print(f"Saving model with best epoch validation mAP_0.5:0.95: {epoch_validation_metric} (at epoch {epoch})")
    torch.save(model.state_dict(), 'best.pt')


class Trainer:
    def __init__(self, model, device, optimizer, train_loader, epochs, validation_loader=None):
        self._model = model
        self._device = device
        self._optimizer = optimizer
        self._train_loader = train_loader
        self._epochs = epochs
        self._validation_loader = validation_loader
        self._validate = self._validation_loader is not None

    def _train_one_epoch(self, epoch):
        self._model.train()

        batches_total_losses = []
        batches_loss_classifier = []
        batches_loss_rpn_box_reg = []
        batches_loss_objectness = []
        batches_loss_box_reg = []

        progress_bar = tqdm(self._train_loader,
                            total=len(self._train_loader), unit='batch', desc=f"Epoch: {epoch}", colour='blue')

        for batch, (images, targets) in enumerate(self._train_loader):
            images = list(image.to(self._device) for image in images)
            targets = [{k: v.to(self._device) for k, v in t.items()} for t in targets]

            batch_loss = self._model(images, targets)

            batch_total_loss = sum(loss for loss in batch_loss.values())
            batch_total_loss_value = batch_total_loss.item()

            self._optimizer.zero_grad()
            batch_total_loss.backward()
            self._optimizer.step()

            batch_loss_classifier = batch_loss['loss_classifier'].item()
            batch_loss_rpn_box_reg = batch_loss['loss_rpn_box_reg'].item()
            batch_loss_objectness = batch_loss['loss_objectness'].item()
            batch_loss_box_reg = batch_loss['loss_box_reg'].item()

            batches_total_losses.append(batch_total_loss_value)
            batches_loss_classifier.append(batch_loss_classifier)
            batches_loss_rpn_box_reg.append(batch_loss_rpn_box_reg)
            batches_loss_objectness.append(batch_loss_objectness)
            batches_loss_box_reg.append(batch_loss_box_reg)

            progress_bar.set_description(
                desc=f"Training epoch {epoch}, Batch {batch + 1} loss {batch_total_loss_value:.4f}")
            progress_bar.update(1)

        learn_rates = [param['lr'] for param in self._optimizer.param_groups]

        total_loss = sum(loss for loss in batches_total_losses) / len(batches_total_losses)
        total_loss_classifier = sum(loss for loss in batches_loss_classifier) / len(batches_loss_classifier)
        total_loss_rpn_box_reg = sum(loss for loss in batches_loss_rpn_box_reg) / len(batches_loss_rpn_box_reg)
        total_loss_objectness = sum(loss for loss in batches_loss_objectness) / len(batches_loss_objectness)
        total_loss_box_reg = sum(loss for loss in batches_loss_box_reg) / len(batches_loss_box_reg)

        train_data = {
            "train/loss": total_loss,
            "train/loss_classifier": total_loss_classifier,
            "train/loss_rpn_box_reg": total_loss_rpn_box_reg,
            "train/loss_objectness": total_loss_objectness,
            "train/loss_box_reg": total_loss_box_reg,
            "learn_rates": learn_rates,
        }

        progress_bar.close()

        print(f"Training epoch {epoch} finished. Epoch training loss: {total_loss:.4f}")

        return train_data, batches_total_losses

    def _validate_one_epoch(self, epoch):
        self._model.eval()

        batches_map = []
        all_targets = []
        all_predictions = []

        progress_bar = tqdm(self._validation_loader,
                            total=len(self._validation_loader), unit='batch', desc=f"Epoch: {epoch}", colour='red')

        for batch, (images, targets) in enumerate(self._validation_loader):
            images = list(image.to(self._device) for image in images)

            with torch.no_grad():
                predictions = self._model(images, targets)

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

        val_data = {
            "metrics/mAP_0.5": epoch_map_50,
            "metrics/mAP_0.5:0.95": epoch_map_50_95
        }

        progress_bar.close()

        print(f"Validating epoch {epoch} finished. Epoch validation mAP_0.5:0.95: {epoch_map_50_95:.4f}")

        return val_data, batches_map

    def train(self):
        epoch_train_losses = []
        epoch_validation_metrics = []
        batch_train_losses = []
        batch_validation_metrics = []

        best_epoch_validation_metric = None
        best_epoch_loss = None

        for epoch in range(1, self._epochs + 1):

            log_data = {'epoch': epoch}

            train_data, batches_total_losses = self._train_one_epoch(epoch)

            epoch_train_losses.append(train_data['train/loss'])
            batch_train_losses += batches_total_losses

            for i, learn_rate in enumerate(train_data['learn_rates']):
                log_data[f'x/lr_{i}'] = learn_rate

            log_data.update(train_data)

            if self._validate:
                val_data, batches_map = self._validate_one_epoch(epoch)

                epoch_validation_metrics.append(val_data['metrics/mAP_0.5:0.95'])
                batch_validation_metrics += batches_map

                log_data.update(val_data)

                if best_epoch_validation_metric is None or best_epoch_validation_metric < val_data[
                    'metrics/mAP_0.5:0.95']:
                    best_epoch_validation_metric = val_data['metrics/mAP_0.5:0.95']
                    _save_model(self._model, val_data['metrics/mAP_0.5:0.95'], epoch)

            else:
                if best_epoch_loss is None or train_data['train/loss'] < best_epoch_loss:
                    best_epoch_loss = train_data['train/loss']
                    _save_model(self._model, train_data['train/loss'], epoch)

            wandb.log(log_data)

        return epoch_train_losses, batch_train_losses, epoch_validation_metrics, batch_validation_metrics
