import torch
import torch.nn.functional as F
import os
import PIL
import click
import torch.nn as nn
import torch.optim as optim
from diffusers import AutoencoderKL
from tqdm import tqdm
import matplotlib.pyplot as plt
import wandb

from datasets.dutils import get_vae_dataloaders


def _imshow_chw(ax, t):
    t = t.detach().float().cpu().clamp(0, 1)
    if t.shape[0] == 1:
        ax.imshow(t[0].numpy(), cmap='gray', vmin=0, vmax=1)
    else:
        ax.imshow(t.permute(1, 2, 0).numpy())


def plot_samples(images, encoded, reconstructed_images, wandb_step=None, wandb_run=None):
    log_dict = {}
    for j in range(4):
        fig, ax = plt.subplots(1, 3)
        # Visualize first 3 channels of latent (or fewer if latent_channels < 3)
        n_channels = min(3, encoded[j].shape[0])
        _imshow_chw(ax[0], encoded[j][:n_channels])
        _imshow_chw(ax[1], images[j])
        _imshow_chw(ax[2], reconstructed_images[j])
        fig.savefig(f'sample_{j}.png')
        if wandb_run is not None:
            log_dict[f'recon/sample_{j}'] = wandb.Image(fig)
        plt.close(fig)
    if wandb_run is not None and log_dict:
        wandb_run.log(log_dict, step=wandb_step)
    
def downsample_image(image, factor):
    return F.interpolate(image, scale_factor=factor, mode='bilinear', align_corners=False)

def diffusability_loss(autoencoder, original_image, latent_image, al):
    """Regularizer: decode downsampled latents vs downsampled pixels.

    Decoder output size is fixed by architecture (each upsample ×2). Interpolate
    downsampling uses rounding, so e.g. 25→12 then decode → 48 vs image 50→50.
    Resize reconstruction to the supervision grid before MSE so shapes match.
    """
    reconstruction_loss_fn = nn.MSELoss()
    downsampled_image = downsample_image(original_image, 1/2)
    downsampled_latent = downsample_image(latent_image, 1/2)
    reconstructed_images = autoencoder.decode(downsampled_latent).sample
    if reconstructed_images.shape[-2:] != downsampled_image.shape[-2:]:
        reconstructed_images = F.interpolate(
            reconstructed_images,
            size=downsampled_image.shape[-2:],
            mode='bilinear',
            align_corners=False,
        )
    reconstruction_loss = reconstruction_loss_fn(reconstructed_images, downsampled_image)
    return reconstruction_loss
        
def loss_fn(autoencoder, images, al, regularizer_type='kl', return_images=False):
    reconstruction_loss_fn = nn.MSELoss()
    latent_dist = autoencoder.encode(images).latent_dist
    encoded = latent_dist.sample()
    reconstructed_images = autoencoder.decode(encoded).sample
    # Compute losses
    reconstruction_loss = reconstruction_loss_fn(reconstructed_images, images)
    if regularizer_type == 'kl':
        regularizer_loss = 0.5 * torch.mean( latent_dist.mean.pow(2) + latent_dist.logvar.exp() - latent_dist.logvar - 1)
    elif regularizer_type == 'downsampling':
        regularizer_loss = diffusability_loss(autoencoder, images, encoded, al)
    else:
        raise ValueError(f'Invalid regularizer type: {regularizer_type}')
    loss =  al * reconstruction_loss + (1-al)* regularizer_loss
    if return_images:
        return loss, reconstruction_loss, regularizer_loss, encoded, reconstructed_images
    else:
        return loss, reconstruction_loss, regularizer_loss

@click.command()
@click.option('--dataset', type=click.Choice(['chr20', 'chr19_chr20']), default='chr20', help='dataset to use. options: chr20, chr19_chr20')
@click.option('--cache_dir', type=str, default='./Chr20_all_cached')
@click.option('--enformer_tsv', type=str, default='./Chr20_all_cached/enfemb_chr20_res1000_dim512.tsv.gz')
@click.option('--starting_res', type=int, default=100, help='Hi-C maps are 100x100; used for block_out_channels.')
@click.option('--num_blocks', type=int, default=2, help='Number of blocks in the encoder and decoder. The final resolution is starting_res // 2**num_blocks.')
@click.option('--latent_channels', type=int, default=4, help='Number of channels in the latent space.')
@click.option('--batch_size', type=int, default=256)
@click.option('--num_workers', type=int, default=0, show_default=True, help='DataLoader workers for train/test loaders.')
@click.option('--num_epochs', type=int, default=50)
@click.option('--regularizer_type', type=click.Choice(['kl', 'downsampling']), default='kl', help='regularizer type. options: kl, downsampling')
@click.option('--al', type=float, default=.99999)
@click.option('--dir', type=str, default=None)
@click.option('--use-wandb', is_flag=True, default=False, help='Log metrics and samples to Weights & Biases.')
@click.option('--wandb-project', type=str, default='Multimodal_genomics', show_default=True)
@click.option('--wandb-run-name', type=str, default=None, help='Optional W&B run name.')
@click.option('--wandb-group', type=str, default='hic_vae', help='Optional W&B group for related runs.')
@click.option('--wandb-entity', type=str, default='vsvivek99-university-of-pennsylvania', help='Optional W&B entity (team or user).')
def train(
    dataset,
    cache_dir,
    enformer_tsv,
    starting_res,
    latent_channels,
    num_blocks,
    batch_size,
    num_workers,
    num_epochs,
    regularizer_type,
    al,
    dir,
    use_wandb,
    wandb_project,
    wandb_run_name,
    wandb_group,
    wandb_entity,
):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if dir is None:
        dir = './chr20_vae'

    wandb_run = None
    if use_wandb:
        wandb_run = wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            group=wandb_group,
            entity=wandb_entity,
            config={
                'dataset': dataset,
                'cache_dir': cache_dir,
                'enformer_tsv': enformer_tsv,
                'starting_res': starting_res,
                'latent_channels': latent_channels,
                'num_blocks': num_blocks,
                'batch_size': batch_size,
                'num_workers': num_workers,
                'num_epochs': num_epochs,
                'al': al,
                'regularizer_type': regularizer_type,
                'wandb_group': wandb_group,
                'output_dir': dir,
                'device': str(device),
            },
        )

    train_loader, test_loader, in_channels = get_vae_dataloaders(
        cache_dir, enformer_tsv, batch_size, num_workers=num_workers
    )

    down_blocks = ('DownEncoderBlock2D',) * num_blocks
    up_blocks = ('UpDecoderBlock2D',) * num_blocks
    out_channels = [ starting_res//2**i for i in range(num_blocks)]
    autoencoder = AutoencoderKL(in_channels= in_channels, out_channels=in_channels,
                                latent_channels=latent_channels,
                                down_block_types=down_blocks,block_out_channels=out_channels,
                                layers_per_block=2,
                                up_block_types=up_blocks, norm_num_groups=out_channels[-1]).to(device=device)
    
    optimizer = optim.Adam(autoencoder.parameters(), lr=1e-3)

    global_step = 0
    try:
        for epoch in range(num_epochs):
            autoencoder.train()
            total_loss = 0.0
            n_train_batches = 0
            pbar = tqdm(train_loader, total=len(train_loader))
            for images, _ in pbar:
                images = images.to(device)  # Move images to GPU if available

                loss, reconstruction_loss, regularizer_loss = loss_fn(autoencoder, images, al, regularizer_type)
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_train_batches += 1
                global_step += 1
                if wandb_run is not None:
                    wandb_run.log({'train/loss': loss.item(), 'train/reconstruction_loss': reconstruction_loss.item(), 'train/regularizer_loss': regularizer_loss.item()}, step=global_step)
                pbar.set_description(f'Loss : {loss.item() : .5f}')

            train_loss_epoch = total_loss / max(n_train_batches, 1)
            autoencoder.save_pretrained(dir)

            # Eval section
            autoencoder.eval()
            with torch.no_grad():
                total_loss = 0.0
                k = 0
                for images, _ in test_loader:
                    images = images.to(device)

                    loss, reconstruction_loss, regularizer_loss, encoded, reconstructed_images = loss_fn(
                        autoencoder, images, al, regularizer_type, return_images=True
                    )

                    total_loss += loss.item()

                    if k == 0:
                        k += 1
                        plot_samples(
                            images,
                            encoded,
                            reconstructed_images,
                            wandb_step=global_step,
                            wandb_run=wandb_run,
                        )

                test_loss_epoch = total_loss / len(test_loader)
                print(f"Test Loss: {test_loss_epoch:.5f}")
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            'train/loss_epoch': train_loss_epoch,
                            'eval/loss': test_loss_epoch,
                            'eval/reconstruction_loss': reconstruction_loss,
                            'eval/regularizer_loss': regularizer_loss,
                            'epoch': epoch,
                        },
                        step=global_step,
                    )

        print('Saving reconstructed images')
        pbar = tqdm(train_loader, total=len(train_loader))
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
    finally:
        if wandb_run is not None:
            wandb.finish()


if __name__ == '__main__':
    train()
