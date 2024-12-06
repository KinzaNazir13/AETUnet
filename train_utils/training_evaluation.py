import torch
from torch import nn
import train_utils.distributed_utils as utils
from .dice_coefficient_loss import dice_loss, build_target
from sklearn.metrics import accuracy_score
import math
import matplotlib.pyplot as plt
import numpy as np

def criterion(inputs, target, loss_weight=None, num_classes: int = 2, dice: bool = True, ignore_index: int = -100):
    losses = {}
    for name, x in inputs.items():
        
        loss = nn.functional.cross_entropy(x, target, ignore_index=ignore_index, weight=loss_weight)
        if dice is True:
            dice_target = build_target(target, num_classes, ignore_index)
            loss += dice_loss(x, dice_target, multiclass=True, ignore_index=ignore_index)
        losses[name] = loss

    if len(losses) == 1:
        return losses['out']

    return (losses['out1'] + losses['out2'] + losses['out3'] + losses['out4']) / 4


def evaluate(model, data_loader, device, num_classes):
    model.eval()
    confmat = utils.ConfusionMatrix(num_classes)
    dice = utils.DiceCoefficient(num_classes=num_classes, ignore_index=255)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    total_accuracy = 0
    total_samples = 0

    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, 100, header):
            image, target = image.to(device), target.to(device)
            output = model(image)
            if len(output) == 1:
                output = output['out']
            else:
                output = output['out4']

            confmat.update(target.flatten(), output.argmax(1).flatten())
            dice.update(output, target)

            # Calculate accuracy
            preds = output.argmax(dim=1).cpu().numpy()
            labels = target.cpu().numpy()
            total_accuracy += accuracy_score(labels.flatten(), preds.flatten()) * len(labels)
            total_samples += len(labels)
           # Visualize the first image in the batch
            visualize(image[0].cpu(), output[0], target[0])
            break  # Visualize just the first batch
        confmat.reduce_from_all_processes()
        dice.reduce_from_all_processes()

    # Average accuracy over all samples
    avg_accuracy = total_accuracy / total_samples

    return confmat, dice.value.item(), avg_accuracy

import matplotlib.pyplot as plt
import numpy as np

def visualize(image, output, label):
    image_np = np.array(image.permute(1, 2, 0))  # Change channel order for visualization
    output_np = output.argmax(0).cpu().numpy()
    label_np = label.cpu().numpy()

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(image_np)
    ax[0].set_title("Original Image")
    ax[1].imshow(label_np, cmap='gray')
    ax[1].set_title("Ground Truth")
    ax[2].imshow(output_np, cmap='gray')
    ax[2].set_title("Segmented Image")
    
    for a in ax:
        a.axis('off')
    
    plt.show()


def train_one_epoch(model, optimizer, data_loader, device, epoch, num_classes,
                    lr_scheduler, print_freq=10, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    if num_classes == 2:
        
        loss_weight = torch.as_tensor([1.0, 1.0], device=device)
    else:
        loss_weight = None

    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        image, target = image.to(device), target.to(device)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image)
            loss = criterion(output, target, loss_weight, num_classes=num_classes, ignore_index=255)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        lr_scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(loss=loss.item(), lr=lr)

    return metric_logger.meters["loss"].global_avg, lr


def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        # lrf=0.001,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
           
            return warmup_factor * (1 - alpha) + alpha
        else:
            
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9
            # return ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - lrf) + lrf
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)
