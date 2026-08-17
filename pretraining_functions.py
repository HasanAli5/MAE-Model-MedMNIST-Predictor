

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm

def train(mae:nn.Module,device:str,dataloader:DataLoader,optimizer:torch.optim.AdamW,epoch:int,scheduler,grad_accumulation:int=1):
    loss_tally = 0.0
    batches = len(dataloader)
    mae.train()
    with tqdm(dataloader, desc=f"Epoch {epoch+1}: Training",leave=False) as bar:
        for i ,(batch, _) in enumerate(bar):
            batch = batch.to(device,non_blocking=True)

            current_accumulation =  min(grad_accumulation,batches - (i // grad_accumulation) * grad_accumulation)

            # Forward pass
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                _,_,loss = mae(batch)
                
                scaled_loss = loss / current_accumulation 

            # back propagate
            scaled_loss.backward()

            # optimiser step
            if (i+1) % grad_accumulation == 0 or (i+1) == len(bar):
                optimizer.step()
                optimizer.zero_grad()

            scheduler.step()

            loss_tally += loss.item()
            bar.set_postfix({"loss": f"{loss:.4f}","lr": f"{optimizer.param_groups[0]['lr']:.3e}"})

    average_loss = loss_tally/batches
    return average_loss, optimizer.param_groups[0]['lr']

def test(mae:nn.Module,device:str,dataloader:DataLoader,epoch:int,test_type:str="Validation"):
    loss_tally = 0.0
    batches = len(dataloader)
    mae.eval()
    with torch.no_grad(), tqdm(dataloader, desc=(f"Epoch {epoch+1}: "+test_type),leave=False) as bar:
        for batch, _ in bar:
            batch = batch.to(device)

            # Forward pass
            output,loss_mask,loss = mae(batch)
            batch_loss = loss.item()
            
            loss_tally += batch_loss
            bar.set_postfix({"loss": f"{loss:.4f}"})
    average_loss = loss_tally/batches
    return average_loss