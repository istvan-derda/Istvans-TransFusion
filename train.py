import os
import argparse
from tqdm import tqdm
from pyprojroot import here
import numpy as np
import torch

from ddpm import *
from data_make import *


def main(args):
    train_data, test_data = LoadData(
        dataset_name=args.dataset_name, 
        seq_len=args.seq_len)
    
    train_data, test_data = np.asarray(train_data), np.asarray(test_data)

    model = train(
        train_data, 
        beta_schedule=args.beta_schedule,
        objective=args.objective, 
        epochs=args.training_epoch, 
        timesteps=args.timesteps, 
        batch_size=args.batch_size, 
        latent_dim=args.hidden_dim, 
        num_layers=args.num_of_layers, 
        n_heads=args.n_head,
        seq_len=args.seq_len)
    
    samples = generate(model, args.seq_len)

    save(samples, args.dataset_name, args.seq_len)


def train(train_data,
          beta_schedule='cosine',
          objective='pred=x0', 
          epochs=5000, 
          timesteps=1000, 
          batch_size=256, 
          latent_dim=256, 
          num_layers=6, 
          n_heads=8,
          seq_len=100):
    
    features = train_data.shape[2]
    train_data = train_data.transpose(0,2,1)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size)  
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = TransEncoder(
        features = features,
        latent_dim = latent_dim,
        num_heads = n_heads,
        num_layers = num_layers
    )

    diffusion = GaussianDiffusion1D(
        model,
        seq_length = seq_len,
        timesteps = timesteps,  
        objective = objective, # pred_x0, pred_v
        loss_type = 'l2',
        beta_schedule = beta_schedule
    )
    
    diffusion = diffusion.to(device)
    lr = 1e-4
    betas = (0.9, 0.99)
    optim = torch.optim.Adam(diffusion.parameters(), lr = lr, betas = betas)
    
    for running_epoch in tqdm(range(epochs)):
        for i, data in enumerate(train_loader):
            data = data.to(device)
            batch_size = data.shape[0]
            optim.zero_grad()
            loss = diffusion(data)
            loss.backward()
            optim.step()

            if i%len(train_loader)==0 and running_epoch%100==0:
                print(f'Epoch: {running_epoch+1}, Loss: {loss.item()}')
    
    return diffusion


def generate(model, len):

    samples = model.sample(len).cpu().numpy().transpose(0, 2, 1)

    return samples


def save(samples, dataset_name, seq_len, directory='saved_files'):
    os.makedirs(here(directory), exist_ok=True)
    np.save(here(f'{directory}/synth-{dataset_name}-{seq_len}.npy'), samples)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--dataset_name',
        choices=['sine','stock','air', 'energy'],
        default='sine',
        type=str)
    
    parser.add_argument(
        '--beta_schedule',
        choices=['cosine','linear', 'quadratic', 'sigmoid'],
        default='cosine',
        type=str)
    
    parser.add_argument(
        '--objective',
        choices=['pred_x0','pred_v', 'pred_noise'],
        default='pred_v',
        type=str)
    
    parser.add_argument(
        '--seq_len',
        help='sequence length',
        default=100,
        type=int)
    
    parser.add_argument(
        '--batch_size',
        help='batch size for the network',
        default=256,
        type=int)
    
    parser.add_argument(
        '--n_head',
        help='number of heads for the attention',
        default=8,
        type=int)
    
    parser.add_argument(
        '--hidden_dim',
        help='number of hidden state',
        default=256,
        type=int)
    
    parser.add_argument(
        '--num_of_layers',
        help='Number of Layers',
        default=6,
        type=int)
    
    parser.add_argument(
        '--training_epoch',
        help='Diffusion Training Epoch',
        default=5000,
        type=int)
    
    parser.add_argument(
        '--timesteps',
        help='Timesteps for Diffusion',
        default=1000,
        type=int)
    
    args = parser.parse_args() 
    
    main(args)