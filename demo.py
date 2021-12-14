import torch
from utils import load_checkpoint
from torchvision.utils import save_image
import torch.optim as optim
import config
from generator_model import Generator
from discriminator_model import Discriminator
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import GetTest


def main():
    disc = Discriminator(in_channels=3).to(config.DEVICE)
    gen = Generator(in_channels=3, features=64).to(config.DEVICE)
    opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999), )
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))

    load_checkpoint(
        config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE,
    )
    load_checkpoint(
        config.CHECKPOINT_DISC, disc, opt_disc, config.LEARNING_RATE,
    )
    test_dataset = GetTest(root_dir="data/test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=3,
        num_workers=config.NUM_WORKERS,
    )
    loop = tqdm(test_loader, leave=True)
    for idx, (x) in enumerate(loop):
        x = x.to(config.DEVICE)
        with torch.no_grad():
            y_fake = gen(x)
            y_fake = y_fake * 0.5 + 0.5  # remove normalization
            save_image(y_fake, config.TEST_DIR + f"/y_gen_{idx}.png")


if __name__ == "__main__":
    main()
