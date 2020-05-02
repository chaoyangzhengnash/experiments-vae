from __future__ import print_function
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import easydict
import os

args = easydict.EasyDict({
        "batch_size": 128,
        "epochs": 10,
        "no_cuda": False,
        "seed": 1,
        "log_interval": 10,
})

args.cuda = not args.no_cuda and torch.cuda.is_available()


torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)


hidden_dim = 20

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, hidden_dim)
        self.fc22 = nn.Linear(400, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                      recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                         os.getcwd() + '/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(epoch)
        with torch.no_grad():
            sample_latent = torch.randn(64, hidden_dim).to(device)
            sample = model.decode(sample_latent).cpu()
            save_image(sample.view(64, 1, 28, 28),
                       os.getcwd() + '/sample_' + str(epoch) + '.png')

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


# display a 2D manifold of the digits
n = 20  # figure with 20x20 digits

# linearly spaced coordinates on the unit square were transformed
# through the inverse CDF (ppf) of the Gaussian to produce values
# of the latent variables z, since the prior of the latent space
# is Gaussian
u_grid = np.dstack(np.meshgrid(np.linspace(0.05, 0.95, n),
                               np.linspace(0.05, 0.95, n)))

rd=np.random.random_sample(hidden_dim-2)
fix_grid=np.tile(rd,n*n).reshape(n,n,-1)
k_grid=np.concatenate((fix_grid,u_grid),axis=2)


z_grid = norm.ppf(k_grid)
x_decoded = model.decode(torch.from_numpy(z_grid.reshape(n*n, hidden_dim)).float().cuda())
x_decoded = x_decoded.reshape(n, n, 28, 28)



save_image(x_decoded.view(n*n, 1, 28, 28), nrow = n,
            fp = os.getcwd() + '/latent_effect'  + str(args.epochs) +'hidd'+str(hidden_dim) +'.png')

"""CNN layer VAE. Change last two dimension of hidden state. to check the difference."""

from __future__ import print_function
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import easydict
import os

args = easydict.EasyDict({
        "batch_size": 1000, 
        "epochs": 30,
        "no_cuda": False, 
        "seed": 1,
        "log_interval": 10,
        "hidden_dim" : 3
})

args.cuda = not args.no_cuda and torch.cuda.is_available()


torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)



class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

        )

        self.fc11 = nn.Linear(128 * 7 * 7, args.hidden_dim)
        self.fc12 = nn.Linear(128 * 7 * 7, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, 128 * 7 * 7)

        self.deconv_block = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def encode(self, x):
        h1 = self.conv_block(x)
        h1 = h1.view(args.batch_size,-1)
        return self.fc11(h1), self.fc12(h1)

    def reparameterize(self, mu, logvar): 
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)  
        return mu + eps*std 

        return z
    
    def decode(self, z): 
        h3 = self.fc2(z).view(-1,128,7,7) # modify shape if you change kernel size
        return self.deconv_block(h3)  

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar) 
        return self.decode(z), mu, logvar


model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum') 

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())  

    return BCE + KLD


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                      recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                         os.getcwd() + '/reconstruction_' + str(epoch) + '.png', nrow=n)
                
    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(epoch)
        with torch.no_grad():
            sample_latent = torch.randn(64, args.hidden_dim).to(device)  
            sample = model.decode(sample_latent).cpu() 
            save_image(sample.view(64, 1, 28, 28), 
                       os.getcwd() + '/sample_' + str(epoch) + '.png')

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

hidden_dim=args.hidden_dim

# display a 2D manifold of the digits
n = 20  # figure with 20x20 digits

# linearly spaced coordinates on the unit square were transformed
# through the inverse CDF (ppf) of the Gaussian to produce values
# of the latent variables z, since the prior of the latent space
# is Gaussian
u_grid = np.dstack(np.meshgrid(np.linspace(0.05, 0.95, n),
                               np.linspace(0.05, 0.95, n)))

rd=np.random.random_sample(hidden_dim-2)
fix_grid=np.tile(rd,n*n).reshape(n,n,-1)
k_grid=np.concatenate((fix_grid,u_grid),axis=2)


z_grid = norm.ppf(k_grid)
x_decoded = model.decode(torch.from_numpy(z_grid.reshape(n*n, hidden_dim)).float().cuda())
x_decoded = x_decoded.reshape(n, n, 28, 28)



save_image(x_decoded.view(n*n, 1, 28, 28), nrow = n,
            fp = os.getcwd() + '/latent_effect_CNN_epoch'  + str(args.epochs) +'hidd'+str(hidden_dim) +'.png')



from google.colab import drive
drive.mount('/content/drive')

#3 layers CNN

args = easydict.EasyDict({
        "batch_size": 1000, 
        "epochs": 20,
        "no_cuda": False, 
        "seed": 1,
        "log_interval": 10,
        "hidden_dim" : 10
})

args.cuda = not args.no_cuda and torch.cuda.is_available()


torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)



class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

        )

        self.fc11 = nn.Linear(256 * 4 * 4, args.hidden_dim)
        self.fc12 = nn.Linear(256 * 4 * 4, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, 256 * 4 * 4)

        self.deconv_block = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def encode(self, x):
        h1 = self.conv_block(x)
        h1 = h1.view(args.batch_size,-1)
        return self.fc11(h1), self.fc12(h1)

    def reparameterize(self, mu, logvar): 
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)  
        return mu + eps*std 

        return z
        
    def decode(self, z): 
        h3 = self.fc2(z).view(-1,256,4,4) # modify shape if you change kernel size
        return self.deconv_block(h3)  

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar) 
        return self.decode(z), mu, logvar


model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum') 

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())  

    return BCE + KLD


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                      recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                         os.getcwd() + '/reconstruction_' + str(epoch) + '.png', nrow=n)
                
    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(epoch)
        with torch.no_grad():
            sample_latent = torch.randn(64, args.hidden_dim).to(device)  
            sample = model.decode(sample_latent).cpu() 
            save_image(sample.view(64, 1, 28, 28), 
                       os.getcwd() + '/sample_' + str(epoch) + '.png')

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

hidden_dim=args.hidden_dim

# display a 2D manifold of the digits
n = 20  # figure with 20x20 digits

# linearly spaced coordinates on the unit square were transformed
# through the inverse CDF (ppf) of the Gaussian to produce values
# of the latent variables z, since the prior of the latent space
# is Gaussian
u_grid = np.dstack(np.meshgrid(np.linspace(0.05, 0.95, n),
                               np.linspace(0.05, 0.95, n)))

rd=np.random.random_sample(hidden_dim-2)
fix_grid=np.tile(rd,n*n).reshape(n,n,-1)
k_grid=np.concatenate((fix_grid,u_grid),axis=2)


z_grid = norm.ppf(k_grid)
x_decoded = model.decode(torch.from_numpy(z_grid.reshape(n*n, hidden_dim)).float().cuda())
x_decoded = x_decoded.reshape(n, n, 28, 28)



save_image(x_decoded.view(n*n, 1, 28, 28), nrow = n,
            fp = os.getcwd() + '/latent_effect_CNN_layer3epoch'  + str(args.epochs) +'hidd'+str(hidden_dim) +'.png')

