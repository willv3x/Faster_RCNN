import wandb
from tqdm import tqdm

from job.validator import Validator
from util.utils import save_model


class Trainer:
    def __init__(self):
        self._best_epoch_loss = None

    def train_one_epoch(self, model, device, optimizer, train_loader, epoch):
        model.train()

        batches_total_losses = []
        batches_loss_classifier = []
        batches_loss_rpn_box_reg = []
        batches_loss_objectness = []
        batches_loss_box_reg = []

        progress_bar = tqdm(train_loader,
                            total=len(train_loader), unit='batch', desc=f"Epoch: {epoch}", colour='blue')

        for batch, (images, targets) in enumerate(train_loader):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            batch_loss = model(images, targets)

            batch_total_loss = sum(loss for loss in batch_loss.values())
            batch_total_loss_value = batch_total_loss.item()

            optimizer.zero_grad()
            batch_total_loss.backward()
            optimizer.step()

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
            "train/batches_loss": batches_total_losses
        }

        for i, learn_rate in enumerate([param['lr'] for param in optimizer.param_groups]):
            train_data[f'x/lr_{i}'] = learn_rate

        progress_bar.close()

        print(f"Training epoch {epoch} finished. Epoch training loss: {total_loss:.4f}")

        if self._best_epoch_loss is None or train_data['train/loss'] < self._best_epoch_loss:
            self._best_epoch_loss = train_data['train/loss']
            print(f"Saving model with best epoch train loss: {train_data['train/loss']} (at epoch {epoch})")
            save_model(model, 'best_train_model')

        return train_data

    def train(self, model, epochs, device, optimizer, train_loader, validation_loader=None):
        validator = Validator()
        log = {}

        epoch_train_losses = []
        epoch_validation_metrics = []

        for epoch in range(1, epochs + 1):

            log['epoch'] = epoch

            train_data = self.train_one_epoch(model, device, optimizer, train_loader, epoch)

            epoch_train_losses.append(train_data['train/loss'])

            log.update(train_data)

            if validation_loader is not None:
                validation_data = validator.validate_one_epoch(model, device, validation_loader, epoch)

                epoch_validation_metrics.append(validation_data['metrics/mAP_0.5:0.95'])

                log.update(validation_data)

            wandb.log(log)

        return epoch_train_losses, epoch_validation_metrics, log


if __name__ == '__main__':
    print('hi')
