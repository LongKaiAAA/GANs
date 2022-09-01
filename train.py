import os
from load import data_load
from model import Generator, Discriminator
from torch.autograd import Variable
import numpy as np
from torchvision.utils import save_image
import torch
import argparse

def base_parameters():
    """
    :return: 默认参数
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--img_size", type=int, default=256, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    parser.add_argument("--sample_interval", type=int, default=400, help="图像采样间隔")
    parser.add_argument("--input_shape", type=tuple, default=(3, 256, 256), help="输入图像的尺寸")
    parser.add_argument("--input_dim", type=int, default=100, help="生成器输出参数的长度")
    opt = parser.parse_args()
    return opt

# 判断文件夹是否存在，没有创建
os.makedirs("images", exist_ok=True)
os.makedirs("saved_models", exist_ok=True)
# 是否使用GUP
cuda = True if torch.cuda.is_available() else False
# 导入参数
opt = base_parameters()
# 损失函数
adversarial_loss = torch.nn.MSELoss()
# 初始化生成器和判别器
generator = Generator(input_dim=opt.input_dim, img_shape=opt.input_shape)    # 生成器
discriminator = Discriminator(img_shape=opt.input_shape)                     # 判别器

# 加载数据
dataloader = data_load(batch_size=opt.batch_size)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
# 优化器
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))


for epoch in range(opt.n_epochs):
    #print(dataloader)
    for i, img in enumerate(dataloader):
        imgs = img

        valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)

        real_imgs = Variable(imgs.type(Tensor))
        #训练生成器
        optimizer_G.zero_grad()

        # 噪声样本
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

        # 利用噪声生成图像
        gen_imgs = generator(z)

        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)

        g_loss.backward()
        optimizer_G.step()

        #训练判别器
        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        # 真实图片的损失
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        # 虚假图片的损失
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        # 判别器的总的损失，真实图片与虚假图片各取一半
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        # 打印每个step后的损失结果
        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )

        # 统计总共训练的step，每经过opt.sample_interval个step就利用当前的生成器参数进行随机生成并保存结果

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            # 随机每400次生成一张图片
            save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)
     # 保存最近一次epoch的网络权重模型到指定路径下
    torch.save(generator.state_dict(), "saved_models/generator_best.pth")
