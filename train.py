import torch
from utils import save_checkpoint,load_checkpoint,save_some_examples
import torch.nn as nn
import torch.optim as optim
import config
from dataset import MapDataset
from generator_model import Generator
from discriminator_model import Discriminator
from torch.utils.data import DataLoader
from tqdm import tqdm  # for process bar

def train_fn(disc,gen,loader,opt_disc,opt_gen,l1_loss,bce,g_scaler,d_scaler):
    loop = tqdm(loader,leave=True)
    for idx,(x,y) in enumerate(loop):
        x,y = x.to(config.DEVICE),y.to(config.DEVICE)

        #Train Discriminator
        with torch.cuda.amp.autocast(): # 自动混合精度混合的是不同类型的tensor。与gradscaler 搭配使用。可以减少缓存占用，适用于上下文情形，训练时间略有增长
            y_fake = gen(x)
            D_real = disc(x,y)
            D_fake = disc(x,y_fake.detach())
            #detach()返回一个新的tensor，从当前计算图中分离下来的，但是仍指向原变量的存放位置,不同之处只是requires_grad为false，得到的这个tensor永远不需要计算其梯度，不具有grad。
            #即使之后重新将它的requires_grad置为true,它也不会具有梯度grad
            D_real_loss = bce(D_real,torch.ones_like(D_real))
            D_fake = disc(x,y.fake.detach())
            D_fake_loss = bce(D_fake,torch.zeros_like(D_fake))
            D_loss = (D_real_loss+D_fake_loss)/2

        disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train Generator
        with torch.cuda.amp.autocast():
            D_fake = disc(x,y_fake)
            G_fake_loss = bce(D_fake,torch.ones_like(D_fake))
            L1 = l1_loss(y_fake,y)*config.L1_LAMBDA
            G_loss = G_fake_loss + L1

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

def main():
    disc = Discriminator(in_channels=3).to(config.DEVICE)
    gen = Generator(in_channels=3).to(config.DEVICE)
    opt_disc = optim.Adam(disc.parameters(),lr = config.LEARNING_RATE,betas=(0.5,0.999))
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    BCE = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss() # work well with a patch gan

    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_GEN,gen,opt_gen,config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_DISC, gen, opt_disc, config.LEARNING_RATE)

    train_dataset = MapDataset(root_dir="data/maps/train")
    train_loader = DataLoader(train_dataset,batch_size=config.BATCH_SIZE,shuffle=True,num_workers=config.NUM_WORKERS)# workers 是进程数
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()
    val_dataset = MapDataset(root_dir = "data/maps/val")
    val_loader = DataLoader(val_dataset,batch_size=1,shuffle=False)

    for epoch in range(config.NUM_EPOCHS):
        train_fn(disc,gen,train_loader,opt_disc,opt_gen,L1_LOSS,BCE,g_scaler,d_scaler)

        if config.SAVE_MODEL and epoch%5==0:
            save_checkpoint(gen,opt_gen,filename=config.CHECKPOINT_GEN)
            save_checkpoint(disc,opt_disc,filename=config.CHECKPOINT_DISC)

        save_some_examples(gen,val_loader,epoch,folder="evaluation")



if __name__ =="__main__":
    main()