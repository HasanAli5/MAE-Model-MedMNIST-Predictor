

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
from sklearn.metrics import roc_auc_score

def train(predictor:nn.Module,device:str,dataloader:DataLoader,transform,loss_fn:nn.CrossEntropyLoss,optimizer:torch.optim.AdamW,epoch:int,scheduler,grad_accumulation:int=1):
    loss_tally = 0.0
    correct_predictions = 0
    total_samples = 0
    batches = len(dataloader)

    truth_labels = []
    pred_labels = []
    all_probabilities = []

    predictor.train()
    with tqdm(dataloader, desc=f"Epoch {epoch+1}: Training",leave=False) as bar:
        for i, (images, labels) in enumerate(bar):
            images = images.to(device,non_blocking=True)
            labels = labels.to(device,non_blocking=True)
            images = transform(images)
            labels = labels.squeeze(-1)

            # Forward pass
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                output = predictor(images)
                # Do CE loss as test
                loss = loss_fn(output, labels)
                
                scaled_loss = loss / grad_accumulation

            # back propagate
            scaled_loss.backward()

            # optimiser step
            if (i+1) % grad_accumulation == 0 or (i+1) == batches:
                torch.nn.utils.clip_grad_norm_(predictor.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            scheduler.step()

            loss_tally += loss.item() if not torch.isnan(loss) else 0.0

            _, predicted_classes = torch.max(output.float(), dim=1)

            correct_predictions += (predicted_classes == labels).sum().item()
            total_samples += labels.size(0)

            probabilities = torch.softmax(output.float(), dim=1)

            truth_labels.extend(labels.detach().cpu().numpy())
            pred_labels.extend(predicted_classes.detach().cpu().numpy())
            all_probabilities.extend(probabilities.detach().cpu().numpy())

            bar.set_postfix({"loss": f"{loss:.4f}","lr": f"{optimizer.param_groups[0]['lr']:.3e}"})
    average_loss = loss_tally/batches
    auc = roc_auc_score(truth_labels,all_probabilities,multi_class='ovr',average='macro')
    final_accuracy = (correct_predictions / total_samples)
    return average_loss,final_accuracy,auc

def finetune(predictor:nn.Module,device:str,dataloader:DataLoader,transform,loss_fn:nn.CrossEntropyLoss,optimizer:torch.optim.AdamW,epoch:int,scheduler,grad_accumulation:int=1):
    loss_tally = 0.0
    correct_predictions = 0
    total_samples = 0
    batches = len(dataloader)

    truth_labels = []
    pred_labels = []
    all_probabilities = []

    predictor.train()
    with tqdm(dataloader, desc=f"Epoch {epoch+1}: Training",leave=False) as bar:
        for i, (images, labels) in enumerate(bar):
            images = images.to(device,non_blocking=True)
            labels = labels.to(device,non_blocking=True)
            images = transform(images)

            # Forward pass
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                output = predictor(images)
                # Do CE loss as test
                loss = loss_fn(output, labels)
                
                scaled_loss = loss / grad_accumulation

            # back propagate
            scaled_loss.backward()

            # optimiser step
            if (i+1) % grad_accumulation == 0 or (i+1) == batches:
                torch.nn.utils.clip_grad_norm_(predictor.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            scheduler.step()

            loss_tally += loss.item() if not torch.isnan(loss) else 0.0

            discrete_truth_labels = torch.argmax(labels, dim=1) 

            _, predicted_classes = torch.max(output.float(), dim=1)

            correct_predictions += (predicted_classes == discrete_truth_labels).sum().item()
            total_samples += labels.size(0)

            probabilities = torch.softmax(output.float(), dim=1)

            truth_labels.extend(discrete_truth_labels.detach().cpu().numpy())
            pred_labels.extend(predicted_classes.detach().cpu().numpy())
            all_probabilities.extend(probabilities.detach().cpu().numpy())

            bar.set_postfix({"loss": f"{loss:.4f}","lr": f"{optimizer.param_groups[0]['lr']:.3e}"})
    average_loss = loss_tally/batches
    auc = roc_auc_score(truth_labels,all_probabilities,multi_class='ovr',average='macro')
    final_accuracy = (correct_predictions / total_samples)
    return average_loss,final_accuracy,auc

def test(predictor:nn.Module,device:str,dataloader:DataLoader,transform,loss_fn:nn.CrossEntropyLoss,epoch:int,test_type:str="Validation"):
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
            if transform is not None:
                images = transform(images)
            labels = labels.squeeze(-1) 
            
            # Forward pass
            output = predictor(images)

            # Do loss as test
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

