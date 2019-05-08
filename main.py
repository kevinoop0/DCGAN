# -*- coding:utf-8 -*-
import os
import torch
import torchvision as tv
import visdom
from model import NetD,NetG
import numpy as np
from torch.utils.data import DataLoader

class Config(object):
    data_path='./data'
    num_workers = 4
    image_size = 96 # the size of image
    batch_size = 256 # the batch size of dataset
    max_epoch = 50 # the max iteration
    lr = 1e-4 # learning rate of Adam
    beta = 0.5 # the first parameter of optimizer
    gpu = True # use gpu
    nz = 100 # the dim of noise
    ngf = 64 # channels of generator feature
    ndf = 64 # channels of discriminator feature
    save_path = 'imgs/'  # the path of saving image

    vis = True  # use visdom
    plot_every = 10  # every 10 step , visiual once

    d_every = 1  # every 1 step , train discriminator once
    g_every = 5  # every 10 step , train generator once
    netd_path = None  # 'checkpoints/netd_.pth'
    netg_path = None  # 'checkpoints/netg_211.pth'

opt=Config()

def train(**kwargs):
    for k_,v_ in kwargs.items():
        setattr(opt,k_,v_)
    if opt.vis:
        vis= visdom.Visdom(env='GAN')
    device= torch.device('cuda'if opt.gpu else 'cpu')
    transform = tv.transforms.Compose([
        tv.transforms.Resize(opt.image_size),
        tv.transforms.CenterCrop(opt.image_size),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
    dataset = tv.datasets.ImageFolder(opt.data_path,transform=transform)
    dataloader = DataLoader(dataset,batch_size=opt.batch_size,shuffle=True,num_workers=opt.num_workers,drop_last=True)

    netd,netg = NetD(opt),NetG(opt)
    # parallel operations
    # netg = torch.nn.DataParallel(netg, device_ids=[0, 1])
    # netd = torch.nn.DataParallel(netd, device_ids=[0, 1])
    netd.to(device)
    netg.to(device)
    optimizer_g = torch.optim.Adam(netg.parameters(),lr = opt.lr,betas=(opt.beta,0.999))
    optimizer_d = torch.optim.Adam(netd.parameters(),lr = opt.lr,betas=(opt.beta,0.999))
    criterion = torch.nn.BCELoss()

    true_labels = torch.ones(opt.batch_size).to(device)
    fake_labels = torch.zeros(opt.batch_size).to(device)
    noise = torch.randn(opt.batch_size,opt.nz,1,1).to(device)
    for epoch in range(opt.max_epoch):
        for i,(img,_) in enumerate(dataloader):
            real_img = img.to(device)
            bs = len(real_img)
            if (i+1) % opt.d_every == 0:
                optimizer_d.zero_grad()
                output=netd(real_img)
                error_d_real = criterion(output,true_labels)
                error_d_real.backward()
                D_x = output.mean().item()

                fake_img = netg(noise)
                fake_output= netd(fake_img)
                error_d_fake= criterion(fake_output,fake_labels)
                error_d_fake.backward()
                optimizer_d.step()
                D_G_z1 = fake_output.mean().item()
                error_d = error_d_real.item()+error_d_fake.item()

            if (i+1) % opt.g_every ==0:
                optimizer_g.zero_grad()
                fake_img= netg(noise)
                fake_output = netd(fake_img)
                error_g = criterion(fake_output,true_labels)
                D_G_z2 = fake_output.mean().item()
                error_g.backward()
                optimizer_g.step()

            if (i+1) % opt.plot_every ==0:
                fix_fake_imgs= netg(noise)
                scores = netd(fix_fake_imgs).detach()
                indexs = scores.topk(64)[1]
                result = []
                for ii in indexs:
                    result.append(fix_fake_imgs.data[ii].detach().cpu()* 0.5 + 0.5)
                vis.images(torch.stack(result), win='fake_img')
                vis.images(real_img.data.cpu().numpy()[:64] * 0.5 + 0.5, win='real')
                # plot loss curve
                # vis.line(Y=np.array([error_d/bs]),X=np.array([i+1]),win='error_d',update='append')
                # vis.line(Y=np.array([error_g.item()/bs]),X=np.array([i+1]),win='error_g',update='append')
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\t D(x): %.4f\t D(G(z)): %.4f / %.4f' % (
                        epoch, opt.max_epoch, i + 1, len(dataloader), error_d/bs, error_g.item()/bs,D_x, D_G_z1, D_G_z2))


def generate(**kwargs):
    for k_, v_ in kwargs.items():
        setattr(opt, k_, v_)

    device = torch.device('cuda') if opt.gpu else torch.device('cpu')

    netg, netd = NetG(opt).eval(), NetD(opt).eval()
    noises = torch.randn(opt.gen_search_num, opt.nz, 1, 1).normal_(opt.gen_mean, opt.gen_std)
    noises = noises.to(device)

    map_location = lambda storage, loc: storage
    netd.load_state_dict(torch.load(opt.netd_path, map_location=map_location))
    netg.load_state_dict(torch.load(opt.netg_path, map_location=map_location))
    netd.to(device)
    netg.to(device)

    # generate image and score the result
    fake_img = netg(noises)
    scores = netd(fake_img).detach()

    # pick good image
    indexs = scores.topk(opt.gen_num)[1]
    result = []
    for ii in indexs:
        result.append(fake_img.data[ii])
    # save image
    tv.utils.save_image(torch.stack(result), opt.gen_img, normalize=True, range=(-1, 1))

if __name__ == '__main__':
    import fire
    fire.Fire()


