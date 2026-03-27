import torch
import os
import PIL
import click
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from diffusers import AutoencoderKL
from tqdm import tqdm
import matplotlib.pyplot as plt

class CUB_200_2011(torch.utils.data.Dataset):
    def __init__(self, path, split='all', transform=None, **kwargs):
        super().__init__()

        assert split in ('all', 'train_val', 'test')

        self.split = split
        self.transform = transform

        CUB_200_2011_PATH = path
        metadata = torch.load(os.path.join(CUB_200_2011_PATH, 'metadata.pth'), weights_only=True)

        # labels
        self.img_id_to_class_id = metadata['img_id_to_class_id']
        self.class_id_to_class_name = metadata['class_id_to_class_name']
        self.class_name_to_class_id = metadata['class_name_to_class_id']

        # captions
        self.img_id_to_encoded_caps = metadata['img_id_to_encoded_caps']
        self.word_id_to_word = metadata['word_id_to_word']
        self.word_to_word_id = metadata['word_to_word_id']

        name = self.split +'_' if split != 'all' else ''
        imgs_path = os.path.join(CUB_200_2011_PATH, f'imgs_{name}128x128.pth')
        self.img_ids = metadata[f'{name}img_ids']
        self.imgs = torch.load(imgs_path)

    def decode_caption(self, cap):
        if isinstance(cap, torch.Tensor):
            cap = cap.tolist()

        arr = [self.word_id_to_word[word_id] for word_id in cap]
        arr = [a for a in arr if a != '<UNKNOWN>']
        return ' '.join(arr)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = self.imgs[idx]
        if self.transform:
            img = self.transform(img)
        img_id = self.img_ids[idx]
        encoded_caps = self.img_id_to_encoded_caps[img_id]
        captions = [ self.decode_caption(cap) for cap in encoded_caps]
        return img, captions


def plot_samples(images, encoded, reconstructed_images):
    for j in range(4):
        fig, ax = plt.subplots(1,3)
        ax[0].imshow(encoded[j].clamp(0,1).permute(1,2,0).cpu().detach().numpy())
        ax[1].imshow(images[j].clamp(0,1).permute(1,2,0).cpu().detach().numpy())
        ax[2].imshow(reconstructed_images[j].clamp(0,1).permute(1,2,0).cpu().detach().numpy())
                        
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

def get_dataset(name, res):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(res)])
    if name == 'cifar10':
        train_dataset = datasets.CIFAR10(root='./saved_datasets', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root='./saved_datasets', train=False, download=True, transform=transform)
        return train_dataset, test_dataset, 3
    if name == 'cifar100':
        train_dataset = datasets.CIFAR100(root='./saved_datasets', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR100(root='./saved_datasets', train=False, download=True, transform=transform)
        return train_dataset, test_dataset, 3
    elif name == 'mnist':
        train_dataset = datasets.MNIST(root='./saved_datasets', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root='./saved_datasets', train=False, download=True, transform=transform)
        return train_dataset, test_dataset, 1
    elif name == 'cub-200':
        transform = transforms.Lambda(lambda x: (x.float() / 255.) * 2. - 1.)
        train_dataset = CUB_200_2011('/network/rit/dgx/dgx_Yelab/kevin_rojas/datasets/cob-200','train_val', transform=transform)
        test_dataset = CUB_200_2011('/network/rit/dgx/dgx_Yelab/kevin_rojas/datasets/cob-200','all', transform=transform)
        return train_dataset, test_dataset, 3

@click.command()
@click.option('--dataset', type=click.Choice(['cifar10','cifar100','mnist', 'cub-200']), default='cub-200')
@click.option('--starting_res', type=int, default=32)
@click.option('--num_blocks', type=int, default=2)
@click.option('--latent_channels', type=int, default=4)
@click.option('--batch_size', type=int, default=256)
@click.option('--num_epochs', type=int, default=50)
@click.option('--al', type=float, default=.99999)
@click.option('--dir', type=str, default=None)
def train(dataset, starting_res, latent_channels, num_blocks, batch_size, num_epochs, al, dir):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if dir is None:
        dir = f'./{dataset}_vae'
        
    train_dataset, test_dataset, in_channels = get_dataset(dataset, starting_res) 
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

        folder = os.path.join(dir, f'reconstructed_{dataset}')
        os.makedirs(folder, exist_ok=True)
        images_np = (reconstructed_images * 255 ).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()

        for i in range(len(images_np)):
            PIL.Image.fromarray(images_np[i], 'RGB').save(os.path.join(folder, f'{k}.png'))
            k+=1

if __name__ == '__main__':
    train()