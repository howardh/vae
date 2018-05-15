import torch
from torch.autograd import Variable
import numpy as np
import random

from PIL import Image
from tqdm import tqdm

from mnist import MNIST

class Encoder(torch.nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.relu = torch.nn.ReLU()
        self.fc1 = torch.nn.Linear(in_features=784, out_features=400, bias=True)
        self.fc2_mean = torch.nn.Linear(in_features=400, out_features=20, bias=True)
        self.fc2_std = torch.nn.Linear(in_features=400, out_features=20, bias=True)
    
    def forward(self, inputs, sample):
        output = inputs.view(-1,28*28)
        output = self.fc1(output)
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
        self.fc1 = torch.nn.Linear(in_features=20, out_features=400, bias=True)
        self.fc2 = torch.nn.Linear(in_features=400, out_features=784, bias=True)
    
    def forward(self, inputs):
        output = self.fc1(inputs)
        output = self.relu(output)

        output = self.fc2(output)
        output = self.sigmoid(output)

        return output

    def to_js(self):
        fc1 = str(self.fc1.weight.data.cpu().numpy().tolist())
        fc2 = str(self.fc2.weight.data.cpu().numpy().tolist())
        code = """
            var $M = milsushi2;
            var fc1 = $M.jsa2mat(%s);
            var fc2 = $M.jsa2mat(%s);
            fc1 = $M.transpose(fc1);
            fc2 = $M.transpose(fc2);
            sample_input = $M.jsa2mat([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]);
            function relu(input) {
                input._data = input._data.map(function(x){
                    if (x<0) return 0;
                    else     return x;
                })
                return input;
            }
            function sigmoid(input) {
                input._data = input._data.map(function(x){
                    return 1/(1+Math.exp(-x));
                })
                return input;
            }
            function decode(input) {
                var output;
                output = $M.mtimes(input, fc1)
                output = relu(output)
                output = $M.mtimes(output, fc2)
                output = sigmoid(output)
                return output
            }
            function draw(data) {
                var ctx = $('canvas')[0].getContext('2d')
                var img = ctx.getImageData(0,0,28,28)
                data._data.forEach(function(x,i){
                    img.data[i*4+0] = x*255;
                    img.data[i*4+1] = x*255;
                    img.data[i*4+2] = x*255;
                    img.data[i*4+3] = 255;
                });
                ctx.putImageData(img,0,0)
            }
            function getInputFromSliders() {
                var output = new Array(20);
                for (var i = 0; i < 20; i++) {
                    output[i] = parseInt($('#latent'+i)[0].value)
                }
                return $M.jsa2mat([output]);
            }
            function updateImage() {
                var input = getInputFromSliders();
                var output = decode(input);
                draw(output);
            }
            $(function(){
                $('input').on('change',updateImage);
            });
        """%(fc1,fc2)
        return code

class LeNet5(torch.nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.tanh = torch.nn.Tanh()
        self.p0 = torch.nn.ConstantPad2d(2,0)
        self.c1 = torch.nn.Conv2d(in_channels=1,out_channels=6,kernel_size=(5,5))
        self.s2 = torch.nn.AvgPool2d(kernel_size=(2,2))
        self.c3 = torch.nn.Conv2d(in_channels=6,out_channels=16,kernel_size=(5,5))
        self.s4 = torch.nn.AvgPool2d(kernel_size=(2,2))
        self.c5 = torch.nn.Conv2d(in_channels=16,out_channels=120,kernel_size=(5,5))

        self.fc1 = torch.nn.Linear(in_features=120,out_features=84,bias=True) 
        self.fc2_mean = torch.nn.Linear(in_features=84,out_features=20,bias=True)
        self.fc2_std = torch.nn.Linear(in_features=84,out_features=20,bias=True)

    def forward(self, inputs, sample):
        #output = self.conv(inputs)
        output = self.p0(inputs)
        output = output.view(1,1,32,32)
        output = self.c1(output)
        output = self.s2(output)
        output = self.tanh(output)
        output = self.c3(output)
        output = self.s4(output)
        output = self.c5(output)
        output = output.view(-1)

        output = self.fc1(output)
        output = self.tanh(output)

        mean = self.fc2_mean(output)
        logvar = self.fc2_std(output) # log(std^2)
        std = logvar.mul(0.5).exp()

        sample = sample.mul(std).add_(mean)

        return sample, mean, logvar

class ConvEncoder(torch.nn.Module):
    def __init__(self):
        super(ConvEncoder, self).__init__()
        self.relu = torch.nn.LeakyReLU()
        self.c1 = torch.nn.Conv2d(in_channels=1,out_channels=32,kernel_size=(5,5),stride=2,padding=2)
        self.c2 = torch.nn.Conv2d(in_channels=32,out_channels=64,kernel_size=(5,5),stride=2,padding=2)
        self.c3 = torch.nn.Conv2d(in_channels=64,out_channels=128,kernel_size=(3,3),stride=2,padding=0)

        self.fc_mean = torch.nn.Linear(in_features=1152,out_features=20,bias=True)
        self.fc_std = torch.nn.Linear(in_features=1152,out_features=20,bias=True)

    def forward(self, inputs, sample):
        output = inputs.view(-1,1,28,28)
        output = self.c1(output)
        output = self.relu(output)
        output = self.c2(output)
        output = self.relu(output)
        output = self.c3(output)
        output = self.relu(output)
        output = output.view(-1,1152)

        mean = self.fc_mean(output)
        logvar = self.fc_std(output) # log(std^2)
        std = logvar.mul(0.5).exp()

        sample = sample.mul(std).add_(mean)

        return sample, mean, logvar

class ConvDecoder(torch.nn.Module):
    def __init__(self):
        super(ConvDecoder, self).__init__()
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

        #self.fc1 = torch.nn.Linear(in_features=20, out_features=84, bias=True)
        #self.fc2 = torch.nn.Linear(in_features=84, out_features=144, bias=True)

        ## Input: 12*12, output: 16*16
        #self.c1 = torch.nn.ConvTranspose2d(in_channels=1,out_channels=8,kernel_size=5,stride=1)
        ## Input: 16*16, output: 20*20
        #self.c2 = torch.nn.ConvTranspose2d(in_channels=8,out_channels=16,kernel_size=5,stride=1)
        ## Input: 20*20, output: 24*24
        #self.c3 = torch.nn.ConvTranspose2d(in_channels=16,out_channels=8,kernel_size=5,stride=1)
        ## Input: 24*24, output: 28*28
        #self.c4 = torch.nn.ConvTranspose2d(in_channels=8,out_channels=1,kernel_size=5,stride=1)

        self.fc = torch.nn.Linear(in_features=20, out_features=1152, bias=True)
        self.c1 = torch.nn.ConvTranspose2d(in_channels=128,out_channels=64,kernel_size=3,stride=2,padding=0)
        self.c2 = torch.nn.ConvTranspose2d(in_channels=64,out_channels=32,kernel_size=5,stride=2,padding=2,output_padding=1)
        self.c3 = torch.nn.ConvTranspose2d(in_channels=32,out_channels=1,kernel_size=5,stride=2,padding=2,output_padding=1)

    def forward(self, inputs):
        #output = self.fc1(inputs)
        #output = self.relu(output)
        #output = self.fc2(output)
        #output = self.relu(output)
        #output = output.view(1,1,12,12)

        #output = self.c1(output)
        #output = self.relu(output)
        #output = self.c2(output)
        #output = self.relu(output)
        #output = self.c3(output)
        #output = self.relu(output)
        #output = self.c4(output)
        #output = self.sigmoid(output)

        output = self.fc(inputs)
        output = self.relu(output)
        output = output.view(-1,128,3,3)

        output = self.c1(output)
        output = self.relu(output)
        output = self.c2(output)
        output = self.relu(output)
        output = self.c3(output)
        output = self.sigmoid(output)

        return output

def loss(in_img, out_img, mean, logvar):
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
    return loss1+loss2
    
mndata = MNIST('./mnist')
data = mndata.load_training()
test_data = mndata.load_testing()

encoder = Encoder().cuda()
#encoder = ConvEncoder().cuda()
decoder = Decoder().cuda()
#decoder = ConvDecoder().cuda()
#optimizer = torch.optim.SGD(list(encoder.parameters())+list(decoder.parameters()), lr = 0.001, momentum=0.9)
optimizer = torch.optim.Adam(list(encoder.parameters())+list(decoder.parameters()), lr=1e-3)

def save_image(data, file_name):
    data *= 255
    data = data.astype(np.int8)
    img = Image.fromarray(data, mode='L')
    img.save(file_name)

def test():
    total_loss = 0
    for d in tqdm(test_data[0], desc='Testing'):
        img_var = Variable(torch.Tensor(d).float().cuda()/255,
                requires_grad=False).view(28,28)
        sample_var = Variable(torch.zeros([20]).float().cuda(), requires_grad=False)
        o,m,lv = encoder(img_var, sample_var)
        o = decoder(o)
        l = loss(img_var, o, m, lv)
        total_loss += l.data[0]

    #x = decoder.to_js()
    #with open('decoder.js','w') as f:
    #    f.write(x)

    return total_loss

def test_batch():
    batch_size = 100
    total_loss = 0
    pbar = tqdm(total=len(test_data[0]),desc='Testing')
    indices = set(list(range(len(test_data[0]))))
    while len(indices) > 0:
        pbar.update(batch_size)
        batch_indices = random.sample(indices,batch_size)
        indices -= set(batch_indices)
        sample_var = Variable(torch.zeros([batch_size,20]).float().cuda(), requires_grad=False)
        batched_data = torch.zeros([batch_size,1,28,28]) # TODO: Replace with empty after updating pytorch
        for i,di in enumerate(batch_indices):
            batched_data[i,0,:,:] = torch.Tensor(test_data[0][di]).view(1,1,28,28)
        img_var = Variable(batched_data.float().cuda()/255,
                requires_grad=False).view(batch_size,1,28,28)

        o,m,lv = encoder(img_var, sample_var)
        o = decoder(o)
        l = loss(img_var, o, m, lv)
        total_loss += l.data[0]*batch_size

    x = decoder.to_js()
    with open('decoder.js','w') as f:
        f.write(x)

    return total_loss

latents = []
output = []
def save_examples(file_name):
    global latents
    global output
    latents = []
    output = []
    for d in test_data[0][:10]+data[0][:10]:
        img_var = Variable(torch.Tensor(d).float().cuda()/255, requires_grad=False).view(28,28)
        sample_var = Variable(torch.zeros([20]).float().cuda(), requires_grad=False)
        o,m,lv = encoder(img_var, sample_var)
        o = decoder(o)
        latents.append(o)
        output.append(o.data.cpu().numpy().reshape((28,28)))
    test_examples = np.concatenate(output[:10],axis=1)
    train_examples = np.concatenate(output[10:],axis=1)
    all_examples = np.concatenate((test_examples,train_examples),axis=0)
    save_image(all_examples, file_name)

def save_truth(file_name):
    latents = []
    output = []
    for d in test_data[0][:10]+data[0][:10]:
        img_var = Variable(torch.Tensor(d).float().cuda()/255, requires_grad=False).view(28,28)
        output.append(img_var.data.cpu().numpy().reshape((28,28)))
    test_examples = np.concatenate(output[:10],axis=1)
    train_examples = np.concatenate(output[10:],axis=1)
    all_examples = np.concatenate((test_examples,train_examples),axis=0)
    save_image(all_examples, file_name)

#save_truth('output/img-truth.png')

iters = 0
epoch = 60000
batch_size = 100
while True:
    file_name = "output/img-%d.png" % iters
    save_examples(file_name)
    tqdm.write('Saved file: %s' % file_name)
    tl = test_batch()
    tqdm.write('Total loss: %s' % tl)
    iters += 1

    indices = set(list(range(len(data[0]))))
    pbar = tqdm(total=len(indices))
    training_loss = 0
    while len(indices) > 0:
        pbar.update(batch_size)
        batch_indices = random.sample(indices,batch_size)
        indices -= set(batch_indices)
        sample = np.random.normal(0,1,(batch_size,20))
        sample_var = Variable(torch.from_numpy(sample).float().cuda(), requires_grad=False)
        batched_data = torch.zeros([batch_size,1,28,28]) # TODO: Replace with empty after updating pytorch
        for i,di in enumerate(batch_indices):
            batched_data[i,0,:,:] = torch.Tensor(data[0][di]).view(1,1,28,28)
        img_var = Variable(batched_data.float().cuda()/255,
                requires_grad=False).view(batch_size,1,28,28)

        optimizer.zero_grad()
        o,m,lv = encoder(img_var, sample_var)
        o = decoder(o)
        l = loss(img_var, o, m, lv)
        training_loss += l.data[0]

        l.backward()
        optimizer.step()
    tqdm.write('Training loss: %s' % training_loss)

    #for d in tqdm(data[0], desc='Training'):
    #    if iters % epoch == 0:
    #        file_name = "output/img-%d.png" % iters
    #        save_examples(file_name)
    #        tqdm.write('Saved file: %s' % file_name)
    #        #tl = test()
    #        #tqdm.write('Total loss: %s' % tl)
    #    iters += 1
    #    sample = np.random.normal(0,1,(batch_size,20))
    #    img_var = Variable(torch.Tensor(data[0][iters%11]).float().cuda()/255,
    #            requires_grad=False).view(28,28)
    #    #img_var = Variable(torch.Tensor(d).float().cuda()/255,
    #    #        requires_grad=False).view(28,28)
    #    sample_var = Variable(torch.from_numpy(sample).float().cuda(), requires_grad=False)
    #    optimizer.zero_grad()
    #    o,m,lv = encoder(img_var, sample_var)
    #    o = decoder(o)
    #    l = loss(img_var, o, m, lv)

    #    l.backward()
    #    optimizer.step()
