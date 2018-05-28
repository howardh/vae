import torch
from torch.autograd import Variable
import numpy as np
import random
import dill

from PIL import Image
from tqdm import tqdm

LATENT_SIZE = 2

class ConvEncoder(torch.nn.Module):
    def __init__(self):
        super(ConvEncoder, self).__init__()
        self.relu = torch.nn.LeakyReLU()
        self.c1 = torch.nn.Conv2d(in_channels=3,out_channels=16,kernel_size=(8,8),stride=4,padding=(0,2))
        self.c2 = torch.nn.Conv2d(in_channels=16,out_channels=32,kernel_size=(4,4),stride=2,padding=(0,2))
        self.c3 = torch.nn.Conv2d(in_channels=32,out_channels=64,kernel_size=(4,4),stride=2,padding=(0,2))
        self.c4 = torch.nn.Conv2d(in_channels=64,out_channels=128,kernel_size=(4,4),stride=2)

        self.fc1 = torch.nn.Linear(in_features=2048,out_features=1024,bias=True)
        self.fc2_mean = torch.nn.Linear(in_features=1024,out_features=LATENT_SIZE,bias=True)
        self.fc2_std = torch.nn.Linear(in_features=1024,out_features=LATENT_SIZE,bias=True)

    def forward(self, inputs, sample):
        output = inputs.view(-1,3,210,160)
        output = self.c1(output)
        output = self.relu(output)
        output = self.c2(output)
        output = self.relu(output)
        output = self.c3(output)
        output = self.relu(output)
        output = self.c4(output)
        output = self.relu(output)
        output = output.view(-1,2048)

        output = self.fc1(output)
        output = self.relu(output)

        mean = self.fc2_mean(output)
        logvar = self.fc2_std(output) # log(std^2)
        std = logvar.mul(0.5).exp()

        sample = sample.mul(std).add_(mean)

        return sample, mean, logvar

class ConvDecoder(torch.nn.Module):
    def __init__(self):
        super(ConvDecoder, self).__init__()
        self.relu = torch.nn.LeakyReLU()
        self.sigmoid = torch.nn.Sigmoid()

        self.fc1 = torch.nn.Linear(in_features=LATENT_SIZE, out_features=1024, bias=True)
        self.fc2 = torch.nn.Linear(in_features=1024, out_features=2048, bias=True)

        self.c1 = torch.nn.ConvTranspose2d(out_channels=64,in_channels=128,kernel_size=(4,4),stride=2,output_padding=(1,1))
        self.c2 = torch.nn.ConvTranspose2d(out_channels=32,in_channels=64,kernel_size=(4,4),stride=2,padding=(0,2),output_padding=(0,1))
        self.c3 = torch.nn.ConvTranspose2d(out_channels=16,in_channels=32,kernel_size=(4,4),stride=2,padding=(0,2),output_padding=(1,0))
        self.c4 = torch.nn.ConvTranspose2d(out_channels=3, in_channels=16,kernel_size=(8,8),stride=4,padding=(0,2),output_padding=(2,0))

    def forward(self, inputs):
        output = self.fc1(inputs)
        output = self.relu(output)
        output = self.fc2(output)
        output = self.relu(output)
        output = output.view(-1,128,4,4)

        output = self.c1(output)
        output = self.relu(output)
        output = self.c2(output)
        output = self.relu(output)
        output = self.c3(output)
        output = self.relu(output)
        output = self.c4(output)
        output = self.sigmoid(output)

        return output

class ConvDiscriminator(torch.nn.Module):
    def __init__(self):
        super(ConvDiscriminator, self).__init__()
        self.relu = torch.nn.LeakyReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.c1 = torch.nn.Conv2d(in_channels=3,out_channels=16,kernel_size=(8,8),stride=4,padding=(0,2))
        self.c2 = torch.nn.Conv2d(in_channels=16,out_channels=32,kernel_size=(4,4),stride=2,padding=(0,2))
        self.c3 = torch.nn.Conv2d(in_channels=32,out_channels=64,kernel_size=(4,4),stride=2,padding=(0,2))
        self.c4 = torch.nn.Conv2d(in_channels=64,out_channels=128,kernel_size=(4,4),stride=2)

        self.fc1 = torch.nn.Linear(in_features=2048,out_features=1024,bias=True)
        self.fc2 = torch.nn.Linear(in_features=1024,out_features=1,bias=True)

    def forward(self, inputs, sample):
        output = inputs.view(-1,3,210,160)
        output = self.c1(output)
        output = self.relu(output)
        output = self.c2(output)
        output = self.relu(output)
        output = self.c3(output)
        output = self.relu(output)
        output = self.c4(output)
        output = self.relu(output)
        output = output.view(-1,2048)

        output = self.fc1(output)
        output = self.relu(output)
        output = self.fc2(output)
        output = self.sigmoid(output)

        return output

def loss(in_img, out_img, mean, logvar):
    # Reconstruction loss
    loss1 = torch.nn.functional.binary_cross_entropy(out_img.view(-1), in_img.view(-1))
    #return loss1

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
    
def load_training_data():
    with open('atari.pkl', 'rb') as f:
        return dill.load(f)
def load_testing_data():
    with open('atari2.pkl', 'rb') as f:
        return dill.load(f)

data = load_training_data()[:2]*500
test_data = load_testing_data()

#encoder = Encoder().cuda()
encoder = ConvEncoder().cuda()
#decoder = Decoder().cuda()
decoder = ConvDecoder().cuda()
discriminator = ConvDiscriminator().cuda()
optimizer = torch.optim.Adam(list(encoder.parameters())+list(decoder.parameters()), lr=1e-3)
#optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-3)
optimizerD = torch.optim.Adam(discriminstor.parameters(), lr=1e-3)

def save_image(data, file_name):
    data = data.astype(np.int8)
    img = Image.fromarray(data, mode='RGB')
    img.save(file_name)

def np_to_torch(data):
    """Convert to NCHW format"""
    data = torch.from_numpy(data)
    data = data.permute(2,0,1)
    data = data.unsqueeze(0)
    return data

def torch_to_np(data):
    """Convert to NHWC format"""
    data = data.permute(0,2,3,1)
    data = data.cpu().numpy()
    return data

def test_batch():
    batch_size = 100
    total_loss = 0
    pbar = tqdm(total=len(test_data),desc='Testing')
    indices = set(list(range(len(test_data))))
    while len(indices) > 0:
        pbar.update(batch_size)
        batch_indices = random.sample(indices,batch_size)
        indices -= set(batch_indices)
        sample_var = Variable(torch.zeros([batch_size,LATENT_SIZE]).float().cuda(), requires_grad=False)
        batched_data = torch.zeros([batch_size,3,210,160]) # TODO: Replace with empty after updating pytorch
        for i,di in enumerate(batch_indices):
            batched_data[i,:,:,:] = np_to_torch(test_data[di])
        img_var = Variable(batched_data.float().cuda()/255, requires_grad=False)

        o,m,lv = encoder(img_var, sample_var)
        o = decoder(o)
        l = loss(img_var, o, m, lv)
        total_loss += l.data[0]*batch_size

    return total_loss

latents = []
output = []
inputs = []
samples = []
def save_examples(file_name):
    global latents # For debugging purposes
    global output
    global inputs
    global samples
    latents = []
    output = []
    inputs = []
    samples = []
    for d in test_data[:10]+data[:10]:
        img_var = Variable(np_to_torch(d).float().cuda()/255, requires_grad=False)
        sample_var = Variable(torch.zeros([LATENT_SIZE]).float().cuda(), requires_grad=False)
        o,m,lv = encoder(img_var, sample_var)
        inputs.append(img_var)
        samples.append(sample_var)
        latents.append(o)
        o = decoder(o)
        o *= 255
        output.append(torch_to_np(o.data)[0])
    test_examples = np.concatenate(output[:10],axis=1)
    train_examples = np.concatenate(output[10:],axis=1)
    all_examples = np.concatenate((test_examples,train_examples),axis=0)
    save_image(all_examples, file_name)

def save_truth(file_name):
    latents = []
    output = []
    for d in test_data[:10]+data[:10]:
        #img_var = Variable(np_to_torch(d).cuda()/255, requires_grad=False)
        img_var = Variable(np_to_torch(d).float().cuda()/255, requires_grad=False)
        img_var *= 255
        output.append(torch_to_np(img_var.data)[0])
    test_examples = np.concatenate(output[:10],axis=1)
    train_examples = np.concatenate(output[10:],axis=1)
    all_examples = np.concatenate((test_examples,train_examples),axis=0)
    save_image(all_examples, file_name)

save_image(data[0],'atari.png')

save_truth('atari-truth.png')
save_examples('atari-ex.png')

iters = 0
batch_size = 1
while True:
    file_name = "output/img-%d.png" % iters
    save_examples(file_name)
    tqdm.write('Saved file: %s' % file_name)
    tl = test_batch()
    tqdm.write('Test loss: %s' % tl)
    iters += 1

    indices = set(list(range(len(data))))
    pbar = tqdm(total=len(indices))
    training_loss = 0
    while len(indices) > 0:
        pbar.update(batch_size)
        batch_indices = random.sample(indices,batch_size)
        indices -= set(batch_indices)

        sample = np.random.normal(0,1,(batch_size,LATENT_SIZE))
        sample_var = Variable(torch.from_numpy(sample).float().cuda(), requires_grad=False)
        #sample_var = Variable(torch.zeros([batch_size,LATENT_SIZE]).float().cuda(), requires_grad=False)

        batched_data = torch.zeros([batch_size,3,210,160]) # TODO: Replace with empty after updating pytorch
        for i,di in enumerate(batch_indices):
            batched_data[i,:,:,:] = np_to_torch(data[di])
        img_var = Variable(batched_data.float().cuda()/255, requires_grad=False)

        optimizer.zero_grad()
        o,m,lv = encoder(img_var, sample_var)
        o = decoder(o)
        l = loss(img_var, o, m, lv)
        training_loss += l.data[0]

        l.backward()
        optimizer.step()
    tqdm.write('Training loss: %s' % training_loss)
