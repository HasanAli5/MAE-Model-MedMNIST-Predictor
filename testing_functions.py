

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
from sklearn.metrics import roc_auc_score


def test(predictor:nn.Module,device:str,dataloader:DataLoader,loss_fn:nn.CrossEntropyLoss,epoch:int,test_type:str="Validation"):
    loss_tally = 0.0
    correct_predictions = 0
    total_samples = 0
    batches = len(dataloader)

    truth_labels = []
    pred_labels = []
    all_probabilities = []

    predictor.eval()
    with torch.no_grad(), tqdm(dataloader, desc=f"Epoch {epoch+1}: {test_type}",leave=False) as bar:
        for images, labels in bar:
            images = images.to(device,non_blocking=True)
            labels = labels.to(device,non_blocking=True)
            labels = labels.squeeze(-1) 
            
            # Forward pass
            output = predictor(images)

            # Do MSE loss as test
            loss = loss_fn(output, labels)
            batch_loss = loss.item()
            
            loss_tally += batch_loss

            _, predicted_classes = torch.max(output, dim=1)

            correct_predictions += (predicted_classes == labels).sum().item()
            total_samples += labels.size(0)

            probabilities = torch.softmax(output, dim=1)

            truth_labels.extend(labels.cpu().numpy())
            pred_labels.extend(predicted_classes.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

            bar.set_postfix({"loss": f"{loss:.4f}"})
    average_loss = loss_tally/batches
    auc = roc_auc_score(truth_labels,all_probabilities,multi_class='ovr',average='macro')
    final_accuracy = (correct_predictions / total_samples)
    return average_loss,final_accuracy,auc