import argparse
import sys
import yaml
import time

import torch
from torch.autograd import Variable, grad
from torch.utils.data import DataLoader
from torch.optim import Adam

from torchvision.utils import make_grid, save_image

from models.temporal_generator import Model as Gen
from models.temporal_discriminator import Model as Dis
from dataset import MovingMNIST

parser = argparse.ArgumentParser(description="TemporalGAN")
parser.add_argument("--config", default="config.yml", help="Path for the config file")
args = parser.parse_args()

with open(args.config) as f:
    opt = yaml.load(f)

torch.manual_seed(opt['seed'])

def calc_gradient_penalty(netD, real_data, fake_data):
    alpha = torch.rand(opt['batch_size'], 1, 1, 1, 1)
    alpha = alpha.expand_as(real_data)
    alpha = alpha.cuda() if opt['use_cuda'] else alpha
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if opt['use_cuda']:
        interpolates = interpolates.cuda()
    interpolates = Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = grad(outputs=disc_interpolates, inputs=interpolates, grad_outputs=torch.ones(disc_interpolates.size()).cuda() if opt['use_cuda'] else torch.ones(disc_interpolates.size()), create_graph=True, retain_graph=True, only_inputs=True)[0]
    
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) **2).mean() * opt['lambda']
    return gradient_penalty

dset = MovingMNIST(opt['dataset_path'])
denormalizer = dset.denormalize
loader = DataLoader(dset, batch_size=opt["batch_size"], shuffle=True, num_workers=4)
loader_iterator = iter(loader)
data_len = len(loader)

gen_o = opt['models']['generator']
dis_o = opt['models']['discriminator']

gen = Gen(z_slow_dim=gen_o["z_slow_dim"], z_fast_dim=gen_o["z_fast_dim"],
          out_channels=gen_o["out_channels"], 
          bottom_width=gen_o["bottom_width"], conv_ch=gen_o["conv_ch"])

dis = Dis(in_channels=dis_o["in_channels"], mid_ch=dis_o["mid_ch"])

if opt['use_cuda']:
    start = time.time()
    gen.cuda()
    end = time.time()
    print("Generator moved to GPU in {} seconds".format(end-start))
    start = time.time()
    dis.cuda()
    end = time.time() 
    print("Discriminator moved to GPU in {} seconds".format(end-start))

start_epoch = 0 #default starting epoch is 0

if opt['resume'] != "":
    cp_data = torch.load(opt['resume'])
    gen.load_state_dict(cp_data['gen_state_dict'])
    dis.load_state_dict(cp_data['dis_state_dict'])
    start_epoch = cp_data['epoch'] + 1
    print("Loaded saved models")

#Tensors for gradients computation
one = torch.FloatTensor([1])
mone = one * -1
if opt['use_cuda']:
    one = one.cuda()
    mone = mone.cuda()

#Default optimizers
optimizerD = Adam(dis.parameters(), lr=1e-4, betas=(0.5, 0.9))
optimizerG = Adam(gen.parameters(), lr=1e-4, betas=(0.5, 0.9))

gen.train()
dis.train()
for epoch in range(start_epoch, opt['epochs']):
    start_time = time.time()
    
    for p in dis.parameters(): p.requires_grad = True
    
    for i in range(opt['dis_iters']):
        dis.zero_grad()
        
        #retrieve another iterator and start a new epoch if dataset is exhausted
        try:
            real_data = next(loader_iterator)
        except StopIteration:
            loader_iterator = iter(loader)
            break
        
        if opt['use_cuda']:
            real_data = real_data.cuda()
        real_data = Variable(real_data)

        d_real_loss = dis(real_data)
        d_real_loss = d_real_loss.mean()
        d_real_loss.backward(mone)   
    
        z = gen.generate_input(opt["batch_size"])
        if opt['use_cuda']:
            z = z.cuda()
        #Disable computation of gradients in generator thanks to volatile
        z = Variable(z, volatile=True)
    
        fake_data = Variable(gen(z).data)
        d_fake_loss = dis(fake_data)
        d_fake_loss = d_fake_loss.mean()
        d_fake_loss.backward(one)
        
        #wgan-gp gradient penalty computation
        gradient_penalty = calc_gradient_penalty(dis, real_data.data, fake_data.data)
        gradient_penalty.backward()

        loss = d_fake_loss - d_real_loss + gradient_penalty
        
        #Simple visual feedback
        sys.stdout.flush()
        sys.stdout.write("\r" + "Epoch "+ str(epoch) + " | Loss for discriminator: " + str(loss.data[0]))
        sys.stdout.flush()
        optimizerD.step()
    
    ######Generator training######
    for p in dis.parameters():
        p.requires_grad = False
    gen.zero_grad()
    
    z = gen.generate_input(opt['batch_size'])
    if opt['use_cuda']:
        z = z.cuda()
    z = Variable(z)
    fake_data = gen(z)
    gen_loss = dis(fake_data)
    gen_loss = gen_loss.mean()
    gen_loss.backward(mone)
    gen_loss = -gen_loss
    optimizerG.step()
    
    end_time = time.time()
    epoch_time = end_time - start_time
    
    ###Savings###
    if (epoch - start_epoch) % opt['save_every'] == 0:
        print()
        print("Saving for epoch {}. Gen loss: {:.2f} | Dis loss: {:.2f}".format(epoch, gen_loss.data[0], loss.data[0]))
        
        #Save summary image
        samples_path = '.' if opt['samples_path'] == "" else opt['samples_path']
        videos = fake_data.data
        video_seq = videos.view(videos.size(0)*videos.size(1), videos.size(2), videos.size(3), videos.size(4))
        video_grid = make_grid(video_seq, nrow=videos.size(0))
        save_image(denormalizer(video_grid), samples_path+"/"+str(epoch)+".png")

        #Save checkpoint
        base_path = "." if opt['checkpoint_base'] == "" else opt['checkpoint_base']
        cp_data = {'epoch': epoch, "epoch_time":epoch_time, "gen_state_dict":gen.state_dict(),
    "dis_state_dict":dis.state_dict(), "last_generated_videos":fake_data.data, "dis_loss":loss, "gen_loss":gen_loss}
        torch.save(cp_data, base_path+"/"+"checkpoint"+str(epoch)+".pth")
