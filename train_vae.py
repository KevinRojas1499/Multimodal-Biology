import torch
import os
import PIL
import click
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from diffusers import AutoencoderKL
from tqdm import tqdm
import matplotlib.pyplot as plt

from Data_loader_PKL import Chr20MultimodalDataset


class _HicOnly(torch.utils.data.Dataset):
    """Wrap multimodal dataset; VAE trains on Hi-C maps (1, 100, 100) only."""

    def __init__(self, base: Chr20MultimodalDataset):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        hic, *_ = self.base[idx]
        return hic, 0


def get_dataset(cache_dir: str, enformer_tsv: str):
    full = Chr20MultimodalDataset(cache_dir, enformer_tsv)
    wrapped = _HicOnly(full)
    n = len(wrapped)
    train_size = int(0.7 * n)
    val_size = int(0.15 * n)
    test_size = n - train_size - val_size
    generator = torch.Generator().manual_seed(42)
    train_dataset, _val_dataset, test_dataset = random_split(
        wrapped, [train_size, val_size, test_size], generator=generator
    )
    return train_dataset, test_dataset, 1


def _imshow_chw(ax, t):
    t = t.detach().float().cpu().clamp(0, 1)
    if t.shape[0] == 1:
        ax.imshow(t[0].numpy(), cmap='gray', vmin=0, vmax=1)
    else:
        ax.imshow(t.permute(1, 2, 0).numpy())


def plot_samples(images, encoded, reconstructed_images):
    for j in range(4):
        fig, ax = plt.subplots(1, 3)
        ax[0].imshow(encoded[j].clamp(0, 1).permute(1, 2, 0).cpu().detach().numpy())
        _imshow_chw(ax[1], images[j])
        _imshow_chw(ax[2], reconstructed_images[j])
        fig.savefig(f'sample_{j}.png')
        plt.close(fig)
        
def loss_fn(autoencoder, images, al, return_images=False):
    reconstruction_loss_fn = nn.MSELoss()
    latent_dist = autoencoder.encode(images).latent_dist
    encoded = latent_dist.sample()
    reconstructed_images = autoencoder.decode(encoded).sample
    # Compute losses
    reconstruction_loss = reconstruction_loss_fn(reconstructed_images, images)
    kl_loss = 0.5 * torch.mean( latent_dist.mean.pow(2) + latent_dist.logvar.exp() - latent_dist.logvar - 1)
    loss =  al * reconstruction_loss + (1-al)* kl_loss
    if return_images:
        return loss, encoded, reconstructed_images
    else:
        return loss

@click.command()
@click.option('--cache_dir', type=str, default='./Chr20_all_cached')
@click.option('--enformer_tsv', type=str, default='./Chr20_all_cached/enfemb_chr20_res1000_dim512.tsv.gz')
@click.option('--starting_res', type=int, default=100, help='Hi-C maps are 100x100; used for block_out_channels.')
@click.option('--num_blocks', type=int, default=2)
@click.option('--latent_channels', type=int, default=4)
@click.option('--batch_size', type=int, default=256)
@click.option('--num_epochs', type=int, default=50)
@click.option('--al', type=float, default=.99999)
@click.option('--dir', type=str, default=None)
def train(cache_dir, enformer_tsv, starting_res, latent_channels, num_blocks, batch_size, num_epochs, al, dir):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if dir is None:
        dir = './chr20_vae'
        
    train_dataset, test_dataset, in_channels = get_dataset(cache_dir, enformer_tsv)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    down_blocks = ('DownEncoderBlock2D',) * num_blocks
    up_blocks = ('UpDecoderBlock2D',) * num_blocks
    out_channels = [ starting_res//2**i for i in range(num_blocks)]
    autoencoder = AutoencoderKL(in_channels= in_channels, out_channels=in_channels,
                                latent_channels=latent_channels,
                                down_block_types=down_blocks,block_out_channels=out_channels,
                                layers_per_block=2,
                                up_block_types=up_blocks, norm_num_groups=out_channels[-1]).to(device=device)
    
    optimizer = optim.Adam(autoencoder.parameters(), lr=1e-3)

    for epoch in range(num_epochs):
        autoencoder.train()
        total_loss = 0
        pbar = tqdm(train_loader, total=len(train_dataset)//batch_size + 1)
        for images, _ in pbar:
            images = images.to(device)  # Move images to GPU if available
            
            loss = loss_fn(autoencoder, images, al)
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_description(f'Loss : {loss.item() : .5f}')
        
        autoencoder.save_pretrained(dir)
        
        # Eval section
        autoencoder.eval()
        with torch.no_grad():
            total_loss = 0
            k = 0
            for images, _ in test_loader:
                images = images.to(device)
                
                loss, encoded, reconstructed_images = loss_fn(autoencoder, images, al, return_images=True)

                total_loss += loss.item()
                
                if k == 0:
                    k+=1
                    plot_samples(images, encoded, reconstructed_images)
            
            print(f"Test Loss: {total_loss/len(test_loader):.5f}")

    print('Saving reconstructed images')
    pbar = tqdm(train_loader, total=len(train_dataset)//batch_size + 1)
    k = 0
    for images, _ in pbar:
        images = images.to(device)  # Move images to GPU if available
        

        latent_dist = autoencoder.encode(images).latent_dist
        encoded = latent_dist.sample()
        reconstructed_images = autoencoder.decode(encoded).sample

        folder = os.path.join(dir, 'reconstructed_chr20')
        os.makedirs(folder, exist_ok=True)
        recon = reconstructed_images.clamp(0, 1)
        if recon.shape[1] == 1:
            for i in range(recon.shape[0]):
                arr = (recon[i, 0] * 255).byte().cpu().numpy()
                PIL.Image.fromarray(arr, mode='L').save(os.path.join(folder, f'{k}.png'))
                k += 1
        else:
            images_np = (recon * 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
            for i in range(len(images_np)):
                PIL.Image.fromarray(images_np[i], 'RGB').save(os.path.join(folder, f'{k}.png'))
                k += 1

if __name__ == '__main__':
    train()
