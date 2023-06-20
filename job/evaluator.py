import numpy as np
import torch
from torchmetrics.detection import MeanAveragePrecision
from tqdm import tqdm


class Evaluator:
    def plotMetricPerClass(self, metric_per_class, metric_total, metric_name, classes):
        print(f"{metric_name} per class")
        empty_string = ''
        if len(classes) > 2:
            num_hyphens = 52
            print('-'*num_hyphens)
            print(f"|     Class{empty_string:<16} | {metric_name}{empty_string:<19}|")
            print('-'*num_hyphens)
            class_counter = 0
            for i in range(0, len(classes)-1, 1):
                class_counter += 1
                print(f"|{class_counter:<3} | {classes[i+1]:<20} | {np.array(metric_per_class[i]):.4f}{empty_string:<15}|")
            print('-'*num_hyphens)
            print(f"|m{metric_name}{empty_string:<23} | {np.array(metric_total):.4f}{empty_string:<15}|")
        else:
            print('-'*40)
            print(f"|Class{empty_string:<10} | {metric_name}{empty_string:<18}|")
            print('-'*40)
            print(f"|{classes[1]:<15} | {np.array(metric_total):.4f}{empty_string:<15}|")
            print('-'*40)
            print(f"|m{metric_name}{empty_string:<12} | {np.array(metric_total):.4f}{empty_string:<15}|")

    def plotEvaluationMap(self, evaluation_summary, classes):
        print(f"Classes: {classes}")

        print('\n')
        print(f"mAP 50: {np.array(evaluation_summary['map_50']):.4f}")
        print(f"mAP 75: {np.array(evaluation_summary['map_75']):.4f}")
        print(f"mAP 0.5:0.95: {np.array(evaluation_summary['map']):.4f}")
        print('\n')
        print(f"mAP small: {np.array(evaluation_summary['map_small']):.4f}")
        print(f"mAP medium: {np.array(evaluation_summary['map_medium']):.4f}")
        print(f"mAP large: {np.array(evaluation_summary['map_large']):.4f}")
        print('\n')
        print(f"mAR 1: {np.array(evaluation_summary['mar_1']):.4f}")
        print(f"mAR 10: {np.array(evaluation_summary['mar_10']):.4f}")
        print(f"mAR 100: {np.array(evaluation_summary['mar_100']):.4f}")
        print('\n')
        print(f"mAR small: {np.array(evaluation_summary['mar_small']):.4f}")
        print(f"mAR medium: {np.array(evaluation_summary['mar_medium']):.4f}")
        print(f"mAR large: {np.array(evaluation_summary['mar_large']):.4f}")
        print('\n')

        self.plotMetricPerClass(evaluation_summary['map_per_class'], evaluation_summary['map'], 'AP', classes)
        print('\n')
        self.plotMetricPerClass(evaluation_summary['mar_100_per_class'], evaluation_summary['mar_100'], 'AR', classes)

    def evaluate(self, model, device, data_loader):
        model.eval()

        progress_bar = tqdm(data_loader, total=len(data_loader), unit='batch', desc='Evaluating', colour='yellow')

        metrics_per_batch = []
        all_targets = []
        all_predictions = []

        for batch, (images, targets) in enumerate(data_loader):
            images = list(image.to(device) for image in images)

            with torch.no_grad():
                outputs = model(images)

            batch_targets = []
            batch_predictions = []
            for output, target, image in zip(outputs, targets, images):
                target_dict = dict()
                prediction_dict = dict()

                target_dict['boxes'] = target['boxes'].detach().cpu()
                target_dict['labels'] = target['labels'].detach().cpu()

                prediction_dict['boxes'] = output['boxes'].detach().cpu()
                prediction_dict['scores'] = output['scores'].detach().cpu()
                prediction_dict['labels'] = output['labels'].detach().cpu()

                all_targets.append(target_dict)
                all_predictions.append(prediction_dict)

                batch_targets.append(target_dict)
                batch_predictions.append(prediction_dict)

            batch_metric = MeanAveragePrecision()
            batch_metric.update(batch_predictions, batch_targets)
            batch_metric = batch_metric.compute()['map'].item()

            metrics_per_batch.append(batch_metric)

            progress_bar.set_description(desc=f"Evaluating, Batch {batch+1} mAP_0.5:0.95 {batch_metric:.4f}")
            progress_bar.update(1)

        test_metric = MeanAveragePrecision(class_metrics=True)
        test_metric.update(all_predictions, all_targets)

        progress_bar.close()

        return test_metric.compute()
