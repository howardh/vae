import torch
import numpy as np
import random
import dill

from PIL import Image
from tqdm import tqdm

LATENT_SIZE = 2048
LATENT_SIZE = 20

class ConvEncoder(torch.nn.Module):
    def __init__(self):
        super(ConvEncoder, self).__init__()
        self.relu = torch.nn.LeakyReLU()
        self.c1 = torch.nn.Conv2d(in_channels=9,out_channels=64,kernel_size=(8,8),stride=2,padding=(0,1))
        self.c2 = torch.nn.Conv2d(in_channels=64,out_channels=128,kernel_size=(6,6),stride=2,padding=(1,1))
        self.c3 = torch.nn.Conv2d(in_channels=128,out_channels=128,kernel_size=(6,6),stride=2,padding=(1,1))
        self.c4 = torch.nn.Conv2d(in_channels=128,out_channels=128,kernel_size=(4,4),stride=2)

        self.fc1_mean = torch.nn.Linear(in_features=128*11*8,out_features=6*LATENT_SIZE,bias=True)
        self.fc1_std = torch.nn.Linear(in_features=128*11*8,out_features=6*LATENT_SIZE,bias=True)

    def forward(self, inputs, actions, sample):
        output = inputs.view(-1,9,210,160)
        output = self.c1(output)
        output = self.relu(output)
        output = self.c2(output)
        output = self.relu(output)
        output = self.c3(output)
        output = self.relu(output)
        output = self.c4(output)
        output = self.relu(output)
        output = output.view(-1,128*11*8)

        batch_size = output.size()[0]
        mean = self.fc1_mean(output).view(-1,6,LATENT_SIZE)[range(batch_size),actions,:]
        logvar = self.fc1_std(output).view(-1,6,LATENT_SIZE)[range(batch_size),actions,:] # log(std^2)
        std = logvar.mul(0.5).exp()

        sample = sample.mul(std).add_(mean)

        return sample, mean, logvar

class ConvDecoder(torch.nn.Module):
    def __init__(self):
        super(ConvDecoder, self).__init__()
        self.relu = torch.nn.LeakyReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()

        self.fc = torch.nn.Linear(in_features=LATENT_SIZE, out_features=128*11*8, bias=True)
        self.fc = torch.nn.Linear(in_features=LATENT_SIZE,
                out_features=3*210*160, bias=True)

        #self.c1 = torch.nn.ConvTranspose2d(out_channels=128,in_channels=128,kernel_size=(4,4),stride=2,output_padding=(0,1))
        #self.c2 = torch.nn.ConvTranspose2d(out_channels=128,in_channels=128,kernel_size=(6,6),stride=2,padding=(0,2),output_padding=(0,1))
        #self.c3 = torch.nn.ConvTranspose2d(out_channels=128,in_channels=128,kernel_size=(6,6),stride=2,padding=(2,2),output_padding=(0,1))
        #self.c4 = torch.nn.ConvTranspose2d(out_channels=3, in_channels=128,kernel_size=(8,8),stride=2,padding=(2,2),output_padding=(0,0))
        self.c1 = torch.nn.ConvTranspose2d(out_channels=128,in_channels=128,kernel_size=(4,4),stride=2)
        self.c2 = torch.nn.ConvTranspose2d(out_channels=128,in_channels=128,kernel_size=(6,6),stride=2,padding=(1,1))
        self.c3 = torch.nn.ConvTranspose2d(out_channels=128,in_channels=128,kernel_size=(6,6),stride=2,padding=(1,1))
        self.c4 = torch.nn.ConvTranspose2d(out_channels=3, in_channels=128,kernel_size=(8,8),stride=2,padding=(0,1))

    def forward(self, inputs):
        output = self.fc(inputs)
        output = output.view(-1,3,210,160)
        #output = self.relu(output)
        #output = output.view(-1,128,11,8)

        #output = self.c1(output)
        #output = self.relu(output)
        #output = self.c2(output)
        #output = self.relu(output)
        #output = self.c3(output)
        #output = self.relu(output)
        #output = self.c4(output)
        output = self.tanh(output)

        return output

def loss(in_img, out_img, mean, logvar):
    return torch.nn.functional.mse_loss(out_img.view(-1), in_img.view(-1))
    # Reconstruction loss
    loss1 = torch.nn.functional.binary_cross_entropy(out_img.view(-1), in_img.view(-1))
    return loss1

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
    return loss1+loss2*0.1

def load_training_data():
    with open('atari-Pong-v0.pkl', 'rb') as f:
        return dill.load(f)
def load_testing_data():
    with open('atari-test-Pong-v0.pkl', 'rb') as f:
        return dill.load(f)

data = load_training_data()[:3]*int(1000/3)
test_data = load_testing_data()

encoder = ConvEncoder().cuda()
decoder = ConvDecoder().cuda()
optimizer = torch.optim.Adam(list(encoder.parameters())+list(decoder.parameters()), lr=1e-5)

def save_image(data, file_name):
    data = data.astype(np.int8)
    img = Image.fromarray(data, mode='RGB')
    img.save(file_name)

def np_to_torch(data):
    """Convert to NCHW format"""
    data = torch.tensor(data)
    data = data.permute(2,0,1)
    data = data.unsqueeze(0)
    data = data.float().cuda()/255
    return data

def torch_to_np(data):
    """Convert to NHWC format"""
    data = data.permute(0,2,3,1)
    data = data.cpu().numpy()
    data *= 255
    return data

def normalize_diff(diff):
    diff = diff.copy()
    diff += np.min(diff)
    diff *= 255/max(np.max(diff), 1)
    return diff

def add_diff(data, diff):
    output = data+diff
    output = np.clip(output, 0, 255)
    return output

def test_batch():
    with torch.no_grad():
        batch_size = 50
        total_loss = 0
        pbar = tqdm(total=len(test_data),desc='Testing')
        indices = set(list(range(len(test_data))))
        while len(indices) > 0:
            pbar.update(batch_size)
            batch_indices = random.sample(indices,batch_size)
            indices -= set(batch_indices)
            sample = torch.zeros([batch_size,LATENT_SIZE]).float().cuda()
            batched_input = torch.empty([batch_size,9,210,160]).cuda()
            actions = []
            batched_output = torch.empty([batch_size,3,210,160]).cuda()
            for i,di in enumerate(batch_indices):
                batched_input[i,:,:,:] = np_to_torch(np.concatenate(test_data[di]['observations'][:-1], axis=2))
                batched_output[i,:,:,:] = np_to_torch(np.array(test_data[di]['observations'][-1], dtype=np.float)-test_data[di]['observations'][-2])
                actions.append(test_data[di]['actions'][-1])

            o,m,lv = encoder(batched_input, actions, sample)
            o = decoder(o)
            l = loss(batched_output, o, m, lv)
            total_loss += l.item()*batch_size

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
    outputs = []
    inputs = []
    samples = []
    empty = np.zeros([210,160,3])
    with torch.no_grad():
        for i,d in enumerate(test_data[:3]+data[:3]):
            obs = d['observations']
            actions = d['actions']

            input_img = np_to_torch(np.concatenate(obs[:-1],axis=2))
            inputs += list(obs[:-1])
            sample = torch.zeros([LATENT_SIZE]).float().cuda()
            o,m,lv = encoder(input_img, [actions[-1]], sample)
            o = decoder(o)

            outputs.append(normalize_diff(np.array(obs[-1], dtype=np.float)-obs[-2]))
            outputs.append(normalize_diff(torch_to_np(o)[0]))
            outputs.append(add_diff(obs[-2], torch_to_np(o)[0]))

    test_inputs = np.concatenate(inputs[:9],axis=1)
    test_outputs = np.concatenate(outputs[:9],axis=1)
    train_inputs = np.concatenate(inputs[9:],axis=1)
    train_outputs = np.concatenate(outputs[9:],axis=1)
    all_examples = np.concatenate((test_inputs,test_outputs,train_outputs,train_inputs),axis=0)
    save_image(all_examples, file_name)

def save_truth(file_name):
    latents = []
    output = []
    empty = np.zeros([210,160,3])
    with torch.no_grad():
        for d in test_data[:10]+data[:10]:
            img = np_to_torch(d)
            output.append(torch_to_np(img.data)[0])
    test_examples = np.concatenate(output[:10],axis=1)
    train_examples = np.concatenate(output[10:],axis=1)
    all_examples = np.concatenate((test_examples,train_examples),axis=0)
    save_image(all_examples, file_name)

#save_image(data[0],'atari.png')

#save_truth('atari-truth.png')
save_examples('atari-ex.png')

iters = 0
batch_size = 1
while True:
    file_name = "output/img-%d.png" % iters
    save_examples(file_name)
    print('Saved file: %s' % file_name)
    tl = test_batch()
    print('Test loss: %s' % tl)
    iters += 1

    indices = set(list(range(len(data))))
    pbar = tqdm(total=len(indices), desc='Training')
    training_loss = 0
    while len(indices) > 0:
        pbar.update(batch_size)
        batch_indices = random.sample(indices,batch_size)
        indices -= set(batch_indices)

        sample = np.random.normal(0,1,(batch_size,LATENT_SIZE))
        sample = torch.tensor(sample, requires_grad=False).float().cuda()
        sample = torch.zeros([batch_size,LATENT_SIZE], requires_grad=False).float().cuda()

        batched_input = torch.empty([batch_size,9,210,160]).cuda()
        batched_output = torch.empty([batch_size,3,210,160]).cuda()
        actions = []
        for i,di in enumerate(batch_indices):
            batched_input[i,:,:,:] = np_to_torch(np.concatenate(data[di]['observations'][:-1], axis=2))
            batched_output[i,:,:,:] = np_to_torch(np.array(data[di]['observations'][-1], dtype=np.float)-data[di]['observations'][-2])
            actions.append(data[di]['actions'][-1])

        optimizer.zero_grad()
        o,m,lv = encoder(batched_input, actions, sample)
        o = decoder(o)
        l = loss(batched_output, o, m, lv)
        training_loss += l.item()

        l.backward()
        optimizer.step()
    pbar.close()
    print('Training loss: %s' % training_loss)

with open('encoder-weights.pt', 'wb') as f:
    torch.save(encoder.state_dict(), f)
