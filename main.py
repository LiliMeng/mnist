from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from utils import weight_init
from utils import save_checkpoint


# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

if args.cuda:
    torch.cuda.manual_seed(args.seed)


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)


def conv( inp_chl, out_chl, ker_size = 3, stride = 1, padding = 1 ):
    return nn.Sequential(
        nn.Conv2d( inp_chl, out_chl, ker_size, stride = stride, padding = padding ),
        nn.BatchNorm2d( out_chl ),
        nn.ReLU( True ),
        )

def tconv( inp_chl, out_chl, ker_size = 4, stride = 2, padding = 1 ):
    return nn.Sequential(
        nn.ConvTranspose2d( inp_chl, out_chl, ker_size, stride = stride, padding = padding ),
        nn.BatchNorm2d( out_chl ),
        nn.ReLU( True ),
        )

class Classification_Net( nn.Module ):
    def __init__(self):
        super().__init__()
        self.conv1 = conv( 1, 32 )
        self.conv2 = conv( 32, 32 )
        self.conv3 = conv( 32, 64, stride = 2 )
        self.conv4 = conv( 64, 64 )
        self.conv5 = conv( 64, 128, stride = 2 )
        self.conv6 = nn.Conv2d( 128, 128, 3, padding = 1 )
        self.fc1 = nn.Linear( 128, 10 )

    def forward( self, x ):
        x0 = self.conv2( self.conv1( x  ) )
        x1 = self.conv4( self.conv3( x0 ) )
        x2 = self.conv6( self.conv5( x1 ) )

        f = x2.mean(3).mean(2)
        pred0 = self.fc1( f )
       
        return x0, x1, x2, pred0
        #return F.log_softmax(pred0, dim=1)

class Regression_Net( nn.Module ):
    def __init__(self):
        super().__init__()
        self.conv7 = conv( 128, 128 )
        self.conv8 = tconv( 128, 64 )
        self.conv9 = conv( 64 + 64, 64 )
        self.conv10 = tconv( 64, 32 )
        self.conv11 = conv( 32 + 32, 32 )
        self.conv12 = nn.Conv2d( 32, 1, 3, padding = 1 )

    def forward( self, x0, x1, x2 ):
        y0 = self.conv8( self.conv7( x2 ) )
        y0 = torch.cat([y0, x1], 1)
        y1 = self.conv10( self.conv9( y0 ) )
        y1 = torch.cat([y1, x0], 1)
        mask = nn.Sigmoid() (self.conv12( self.conv11( y1 ) ))

        return mask

class Mask_Net( nn.Module ):
    def __init__( self ):
        super(Mask_Net, self).__init__()
        
        self.cls = Classification_Net()
        self.reg = Regression_Net()

        for module in self.children():
            module.apply(weight_init)

    def forward(self, x, stage = 1):
        x0, x1, x2, pred0 = self.cls( x )

        if stage == 0:
            return pred0, pred0, None

        mask = self.reg( x0, x1, x2 )

        x = x * mask.expand( x.size() )

        x0, x1, x2, pred1 = self.cls( x )

        #return pred0, pred1, mask
        return F.log_softmax(pred1, dim=1), mask 



use_cls_net = False
use_reg_net = False
use_mask_net = True

if use_cls_net == True:
    model = Classification_Net()
elif use_reg_net == True:
    model = Regression_Net()
elif use_mask_net == True:
    model = Mask_Net()
else:
    raise Exception("Not imeplemented yet")

if args.cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)


       
def train_reg(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
      
        optimizer.zero_grad()
        if use_cls_net == True:
            x0, x1, x2, pred0 = model(data)
            output = F.log_softmax(pred0, dim=1)
        elif use_reg_net == True:
            restored_model = torch.load('./checkpoint.pth.tar')
            x0, x1, x2, pred0 = restored_model(data)
            output = model(x0, x1, x2)
        elif use_mask_net == True:
            pretrained_model = torch.load('./saved_checkpoints/checkpoint.pth.tar')['model']
            #print("pretrained_model")
            #print(pretrained_model)
            new_model_dict = model.state_dict()
            for k, v in pretrained_model:
                new_model_dict [k] = v
            model.load_state_dict(new_model_dict)
            output, mask = model(data)
        else:
            raise Exception("Not implemented yet")

        
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))

    

def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        if use_cls_net == True:
            x0, x1, x2, pred0 = model(data)
            output = F.log_softmax(pred0, dim=1)
        elif use_reg_net == True:
            restored_model = torch.load('./checkpoint.pth.tar')
            x0, x1, x2, pred0 = restored_model(data)
            output = model(x0, x1, x2)
        else:
            raise Exception("Not implemented yet")

        #for pic, img in zip(mask.mask, batch[0]):
        # img = data[0]
        # img = img.type(torch.FloatTensor).data
        # img = img.numpy()
        # img = img.transpose( 1, 2, 0 )
        # #mean = np.array([x/255.0 for x in [125.3, 123.0, 113.9]])
        # #std  = np.array([x/255.0 for x in [63.0, 62.1, 66.7]])
        # #img = (img * std + mean) * 255
        # img = img.astype(np.uint8)

        # pic = pic[0].type( torch.FloatTensor )
        # print(pic)
        # print(pic.max())
        # print(pic.min())
        # print(pic.mean())
        # pic = pic.data.numpy()
        # pic -= pic.min()
        # pic /= pic.max()
        # pic *= 255
        # pic = pic.astype( np.uint8 )
        # pic = cv2.applyColorMap( pic, cv2.COLORMAP_JET )
        # cv2.imshow('x', pic)
        # cv2.imshow('y', img)
        # cv2.waitKey(0)
        test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


for epoch in range(1, args.epochs + 1):
    train_reg(epoch)
   # test()
if use_cls_net == True:
    save_checkpoint({
                'epoch': epoch,
                'model': model.state_dict(),
            }, is_best=False, save_folder="saved_checkpoints" , filename='checkpoint.pth.tar')
else:
    print("No need to store model")
#torch.save(model, 'checkpoint.pth.tar')
