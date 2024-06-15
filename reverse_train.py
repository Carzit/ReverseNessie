import os
import datetime
import logging

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from reverse_net import *
from reverse_loss import *

from tqdm import tqdm
from utils.save_and_load import save

def train(
        epoches:int, 
        optimizer:torch.optim.Optimizer,
        encoder:DistributionEncoder_NegativeBinomial, 
        decoder:DistributionDecoder_NegativeBinomial,
        encoder_loss_fn:EncoderLoss, 
        decoder_loss_fn:DecoderLoss, 
        train_generator:DataLoader, 
        val_generator:DataLoader,
        *,
        lr_scheduler:torch.optim.lr_scheduler.LRScheduler=None,
        hparams:dict=None,
        log_dir:str = r"log",
        check_per_batch:int=0,
        print_per_epoch:int=1,
        save_per_epoch:int=1,
        save_dir:str=os.curdir,
        save_name:str="model",
        save_format:str="pt",
        device:torch.device=torch.device('cpu'))->torch.nn.Module:
    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    writer = SummaryWriter(os.path.join(log_dir, "TRAIN"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
    encoder = encoder.to(device=device)
    decoder = decoder.to(device=device)

    # Train
    for epoch in range(epoches):
        
        # Train one Epoch
        encoder.train()
        decoder.train()
        for batch, (cme_params, x_seq, p_seq) in enumerate(tqdm(train_generator)):
            optimizer.zero_grad()

            cme_params = cme_params.to(device=device)
            x_seq = x_seq.to(device=device)
            p_seq = p_seq.to(device=device)

            w, r, p = encoder(p_seq)
            encoder_loss = encoder_loss_fn(w, r, p, x_seq, p_seq)
            cme_pred = decoder(w, p ,r)
            decoder_loss = decoder_loss_fn(cme_pred, cme_params)

            train_loss = decoder_loss + encoder_loss
            
            train_loss.backward()
            optimizer.step()
            
            if check_per_batch:
                if batch % check_per_batch == 0:
                    print(f"Batch: {batch} \nweights: {w} \nr: {r} \np: {p} \n cme_params:{cme_pred}")
                    

        # Record Train Loss Scalar
        writer.add_scalars("Train Loss", 
                           {"trainloss":train_loss.item(), "encoderloss":encoder_loss.item(), "decoderloss":decoder_loss.item()}, 
                           epoch)
        
        # If validation datasets exisit, calculate val loss without recording grad.
        if val_generator:
            encoder.eval() # set eval mode to frozen layers like dropout
            decoder.eval()
            with torch.no_grad(): 
                for batch, (cme_params, x_seq, p_seq) in enumerate(tqdm(val_generator)):
                    cme_params = cme_params.to(device=device)
                    x_seq = x_seq.to(device=device)
                    p_seq = p_seq.to(device=device)

                    w, r, p = encoder(p_seq)
                    encoder_loss = encoder_loss_fn(w, r, p, x_seq, p_seq)
                    cme_pred = decoder(w, p ,r)
                    decoder_loss = decoder_loss_fn(cme_pred, cme_params)

                    val_loss = decoder_loss + encoder_loss
                
                writer.add_scalars("Val Loss", 
                           {"valloss":train_loss.item(), "encoderloss":encoder_loss.item(), "decoderloss":decoder_loss.item()}, 
                           epoch)
                writer.add_scalars("Train-Val Loss", {"Train Loss": train_loss.item(), "Validation Loss": val_loss.item()}, epoch)


        # If learning rate scheduler exisit, update learning rate per epoch.
        writer.add_scalar("Learning Rate", optimizer.param_groups[0]["lr"], epoch)
        if lr_scheduler:
            lr_scheduler.step()
        
        # Flushes the event file to disk
        writer.flush()

        # Specify print_per_epoch = 0 to unable print training information.
        if print_per_epoch:
            if (epoch+1) % print_per_epoch == 0:
                print('Epoch [{}/{}], Train Loss: {:.6f}, Validation Loss: {:.6f}'.format(epoch+1, epoches, train_loss.item(), val_loss.item()))
        
        # Specify save_per_epoch = 0 to unable save model. Only the final model will be saved.
        if save_per_epoch:
            if (epoch+1) % save_per_epoch == 0:
                save(encoder, os.path.join(save_dir, save_name+f"{save_name}_epoch{epoch}_"+"encoder"), save_format)
                save(decoder, os.path.join(save_dir, save_name+f"{save_name}_epoch{epoch}_"+"decoder"), save_format)
        
    writer.close()
    save(encoder, os.path.join(save_dir, save_name+f"{save_name}_final_"+"encoder"), save_format)
    save(decoder, os.path.join(save_dir, save_name+f"{save_name}_final_"+"decoder"), save_format)

if __name__ == "__main__":

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)-5.5s]  %(message)s")
    data