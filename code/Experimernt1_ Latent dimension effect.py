
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
        "epochs": 20,
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

class VAE(nn.Module):
    def __init__(self, hidden):
        super(VAE, self).__init__()

        self.hidden = hidden
        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, self.hidden)
        self.fc22 = nn.Linear(400, self.hidden)
        self.fc3 = nn.Linear(self.hidden, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        #吧logvar变为std
        std = torch.exp(0.5*logvar)
        #sample 的eps 每个数0-1，服从正态分布，20维
        eps = torch.randn_like(std)
        #最终得到的东西，服从的是一个20维的正太分布———使得gradient可导可训练
        return mu + eps*std
        #20维的vector，eps*std,element wise
        #避免采样的过程不可导：eg，直接sample出来的因为是2200维的，没法学 mu和stdstd

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar#return mu 和

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')##又叫deonstrcution loss
    #算原来的image和decode后的距离

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)

    # 希望学出来的 mu and std(分布) 是接近 标准正态分布
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # bce 和kld互相相加，二者其实是互相矛盾的，trade off
    return BCE + KLD

def train(epoch,hidden):

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


def test(epoch,hidden):
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
                if epoch ==20:
                  save_image(comparison.cpu(),
                          os.getcwd() + '/reconstruction_hidden:' + str(hidden) + "_epoch:" +str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

num_hid_dim = 20
hidden = 1
if __name__ == "__main__":
  for hidden in range(1 ,num_hid_dim +1):
   # if (hidden/5).is_integer():
      for epoch in range(1, args.epochs + 1):
          model = VAE(hidden).to(device)
          optimizer = optim.Adam(model.parameters(), lr=1e-3)
          train(epoch,hidden)
          test(epoch,hidden)
          with torch.no_grad():
              sample_latent = torch.randn(64, hidden).to(device)#随机生成六十四二十列的数字()64是batch，变成一个高斯分布
              sample = model.decode(sample_latent).cpu()#decode，把一个20维度变为64*784
              if epoch == 20:
                save_image(sample.view(64, 1, 28, 28),#batch --返回64个图片，1-- 黑白照片
                           os.getcwd() + '/sample_hidden' + str(hidden) + "_epoch:" + str(epoch) + '.png')

#希望最后一维是渐变的过程，最后一个数字每次加 0.1，需要试下hidden dimention
sample_latent.shape
sample_latent[:4]

!zip -r /content/file.zip /content