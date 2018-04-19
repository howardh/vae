import torch
from torch.autograd import Variable
import numpy as np

from tqdm import tqdm

from mnist import MNIST

class Encoder(torch.nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.relu = torch.nn.ReLU()
        self.fc1 = torch.nn.Linear(in_features=784, out_features=500, bias=True)
        self.fc2_mean = torch.nn.Linear(in_features=500, out_features=100, bias=True)
        self.fc2_std = torch.nn.Linear(in_features=500, out_features=100, bias=True)
    
    def forward(self, inputs, sample):
        output = self.fc1(inputs)
        output = self.relu(output)

        mean = self.fc2_mean(output)
        logvar = self.fc2_std(output) # log(std^2)
        std = logvar.mul(0.5).exp()

        sample = sample.mul(std).add_(mean)

        return sample, mean, logvar

class Decoder(torch.nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.fc1 = torch.nn.Linear(in_features=100, out_features=500, bias=True)
        self.fc2 = torch.nn.Linear(in_features=500, out_features=784, bias=True)
    
    def forward(self, inputs):
        output = self.fc1(inputs)
        output = self.relu(output)

        output = self.fc2(output)
        output = self.sigmoid(output)

        return output

def loss(in_img, out_img, mean, logvar):
    # Reconstruction loss
    loss1 = torch.nn.functional.binary_cross_entropy(out_img, in_img)

    # KL-divergence regularization
    # See https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
    # mean of p(z) = 0 = m2
    # std of p(z) = 1 = s2
    # log(1/s1) + (s1^2+mu1^2)/2 - 1/2
    # logvar = log(s^2), log(1/s) = -log(s^2)/2
    # e^logvar = s^2
    loss2 = (-logvar+logvar.exp()+mean.pow(2)-1)/2
    loss2 = torch.mean(loss2)

    # Total loss
    return loss1+loss2
    
mndata = MNIST('./mnist')
data = mndata.load_training()
test_data = mndata.load_testing()

encoder = Encoder()
decoder = Decoder()
optimizer = torch.optim.SGD(list(encoder.parameters())+list(decoder.parameters()), lr = 0.001, momentum=0.9)

def test():
    total_loss = 0
    for d in tqdm(test_data[0], desc='Testing'):
        img_var = Variable(torch.Tensor(d).float()/255, requires_grad=False)
        sample_var = Variable(torch.zeros([100]).float(), requires_grad=False)
        o,m,lv = encoder(img_var, sample_var)
        o = decoder(o)
        l = loss(img_var, o, m, lv)
        total_loss += l.data[0]
    return total_loss

iters = 0
epoch = 1000
for d in tqdm(data[0], desc='Training'):
    iters += 1
    if iters % epoch == 0:
        tl = test()
        tqdm.write('Total loss: %s' % tl)
    sample = np.random.normal(0,1,100)
    img_var = Variable(torch.Tensor(d).float()/255, requires_grad=False)
    sample_var = Variable(torch.from_numpy(sample).float(), requires_grad=False)
    o,m,lv = encoder(img_var, sample_var)
    o = decoder(o)
    l = loss(img_var, o, m, lv)

    optimizer.zero_grad()
    l.backward()
    optimizer.step()
