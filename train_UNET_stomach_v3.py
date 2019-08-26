import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import random
import torchvision.transforms as standard_transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from torch.backends import cudnn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from itertools import product
from sklearn.model_selection import train_test_split
#import transforms as extended_transforms
from PIL import Image, ImageDraw, ImageFont
# from skimage.feature import canny
from misc_v2 import *
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # specify which GPU(s) to be used
def show_mask(image, mask):
    image = np.array(image.convert('L'))   # convert PIL to numpy grayscale
    stitched = np.multiply(image, mask.astype('float'))
    return Image.fromarray(stitched)
def display_images(normalized_inputs, labels, predictions, restore, visualize, save_path, DSCs=None):
    image_display = []
    for idx, (images, dsc) in enumerate(zip(zip(normalized_inputs, labels, predictions), DSCs)):
        if images[0] is None:
            continue
        # images[0] of size (minibatch, 3, 512, 512), images[1] size (minibatch, 512, 512)
        N, size_x, size_y = images[0].size()[0], images[1][0].shape[0], images[1][0].shape[1]
        #size of the image (3*size_x, N*size_y), display (3,N) image, each of size (size_x, size_y)
        result_PIL = Image.new("RGB", (3 * size_x, N * size_y))
        for i in range(N):
            disp_img = []
            disp_img.append(restore(images[0][i]))                           # original image
            disp_img.append(show_mask(restore(images[0][i]), images[1][i]))  # label
            disp_img.append(show_mask(restore(images[0][i]), images[2][i]))  # prediction
            for j in range(3):
                # the given 4-tuple is for defining the position
                result_PIL.paste(disp_img[j], (j * size_x, i * size_y, (j+1) * size_x, (i + 1) * size_y ) )   #coord = (x1, y1, x2, y2)
            if DSCs is not None:
                draw = ImageDraw.Draw(result_PIL)
                font_path = '/mnt/ibrixfs05-Radiomics/Sunan_liver_seg/font/FreeSans.ttf'
                font = ImageFont.truetype(font_path, 25)
                # only display the dice
                dsc_caption = 'liver DSC: {:.3f}, tumor DSC: {:.3f}'.format(dsc[i, -2],dsc[i, -1])
                draw.text((2 * size_x + 30, i * size_y + 10), dsc_caption, font=font, fill='black')
        result_PIL.save(os.path.join(save_path, 'batch_%d_display.png' % idx))   # label image
    print('images saved at:', save_path)
def train(train_loader, net, criterion, optimizer, epoch, config, restore, visualize, device):
    net.train()
    losses = []
    N_count = 0
    train_loss = AverageMeter()
    X_batch_disp, y_batch_disp, y_batch_pred_disp, DSC_batch_disp = [], [], [], []
    for batch_idx, (X, y) in enumerate(train_loader):
        N = X.size(0)
        N_count += N
        X, y = X.squeeze_(0), y.squeeze_(0)
        assert X.size()[2:] == y.size()[1:]
        # check input image size  = label image size
        X, y = Variable(X).to(device), Variable(y).to(device)
        optimizer.zero_grad()
        outputs = net(X)
        # convert probability map => pixels
        #max(1)[1] get the predicted class (num_batch, num_class), squeeze dim of size 1 at 0/1 if applicable
        y_pred = outputs.data.max(1)[1].squeeze_(1).squeeze_(0).cpu().numpy()
        assert outputs.size()[2:] == y.size()[1:]       # check output image dim = mask (label) image dim
        assert outputs.size()[1] == config.num_classes  # check output channel = num of class
        #get the mean of loss
        loss = criterion(outputs, y) / N
        # update loss history
        losses.append(loss.item())
        train_loss.update(loss.item(), N)
        loss.backward()
        optimizer.step()
        # randomly append input image for display
        if random.random() > config.train_img_sample_rate:
            X_batch_disp.append(None)
            y_batch_disp.append(None)
            y_batch_pred_disp.append(None)
        else:
            X_batch_disp.append(X.squeeze_(0).cpu())
            y_batch_disp.append(y.squeeze_(0).cpu().numpy())
            y_batch_pred_disp.append(y_pred)
        # print batch process information
        if (batch_idx + 1) % config.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.2f}'.format(
                epoch + 1, N_count, len(train_loader.dataset),
                100. * (batch_idx + 1) / len(train_loader), loss.item()))
        # over all evaluation
        DSC_batch_disp.append(DSC(y_pred, y.squeeze_(0).cpu().numpy(), config.num_classes))
    # display some training results
    if config.train_save_to_img_file:
        train_save_dir = os.path.join(ckpt_path, exp_name, 'train_epoch{}'.format(epoch + 1))
        #create save path if not exists
        check_mkdir(train_save_dir)
        #restore is transformation, visualize
        display_images(X_batch_disp, y_batch_disp, y_batch_pred_disp, restore, visualize, train_save_dir, DSC_batch_disp)
    return net, optimizer, train_save_dir
def validate(val_loader, net, criterion, optimizer, epoch, config, restore, visualize, device):
    net.eval()
    N_count = 0
    val_loss = AverageMeter()
    X_batch_disp, y_batch_disp, y_batch_pred_disp, DSC_batch_disp = [], [], [], []
    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(val_loader):
            N_count += X.size(0)
            X, y = X.squeeze_(0), y.squeeze_(0)
            X, y = Variable(X).to(device), Variable(y).to(device)
            outputs = net(X)
            # convert probability map => pixels
            y_pred = outputs.data.max(1)[1].squeeze_(1).squeeze_(0).cpu().numpy()
            N = X.size(0)
            loss = criterion(outputs, y).item() / N
            val_loss.update(loss, N)
            # randomly append input image for display
            if random.random() > config.validation_img_sample_rate:
                X_batch_disp.append(None)
                y_batch_disp.append(None)
                y_batch_pred_disp.append(None)
            else:
                X_batch_disp.append(X.squeeze_(0).cpu())
                y_batch_disp.append(y.squeeze_(0).cpu().numpy())
                y_batch_pred_disp.append(y_pred)
            # over all evaluation (using confusion matrix)
            DSC_batch_disp.append(DSC(y_pred, y.squeeze_(0).cpu().numpy(), config.num_classes))
            # print batch process information
            if (batch_idx + 1) % config.log_interval == 0:
                print('valid epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.2f}'.format(
                    epoch + 1, N_count, len(val_loader.dataset), 100. * (batch_idx + 1) / len(val_loader), loss))
        # Save images for display
    validate_save_dir = os.path.join(ckpt_path, exp_name, 'valid_epoch{}'.format(epoch + 1))
    check_mkdir(validate_save_dir)
    print("saving images for display\n")
    display_images(X_batch_disp, y_batch_disp, y_batch_pred_disp, restore, visualize, validate_save_dir, DSC_batch_disp)
    return val_loss.avg
#where the data is
data_path = '/mnt/ibrixfs05-Radiomics/Sunan_liver_seg/LITS_challenge/Train'
#data_path = '/nfs/turbo/umms-ielnaqa/sunan/LITS_challenge/Train/'
# output file directory
#ckpt_path = '/nfs/turbo/umms-ielnaqa/sunan/Segmentation_results/model_ckpt'
ckpt_path = '/mnt/ibrixfs05-Radiomics/Sunan_liver_seg/Segmentation_results/model_ckpt'
exp_name = 'LITS_UNET'
#data_path = 'C:/Users/sunan/Downloads/LITS_challenge/Train/'
#ckpt_path='C:/Users/sunan/Desktop/flux_res/seg'
# load patient image info
patient_image_range = './patient_image_range.pkl'
class Config(object):
    epoch_num = 250
    lr = 1e-3
    batch_size =10
    num_classes = 3    # categories of labels (LITS)
    log_interval = 10
    train_save_to_img_file = True       # save training images
    validation_save_to_img_file = True  # save validation images
    train_img_sample_rate = 0.1        # randomly sample some training results to display
    validation_img_sample_rate = 0.1   # randomly sample some validation results to display
config = Config()
mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# transformation for visualization
visualize = standard_transforms.Compose([
    standard_transforms.Resize(400),
    standard_transforms.CenterCrop(400),
    standard_transforms.ToTensor()
])
# no transformation
no_transform = standard_transforms.ToPILImage()
# Detect devices
use_cuda = torch.cuda.is_available()                   # check if GPU exists
device = torch.device("cuda" if use_cuda else "cpu")   # use CPU or GPU

# data loading parameters
train_params = {'batch_size': config.batch_size, 'shuffle': True, 'num_workers': 0, 'pin_memory': True} if use_cuda else {}
valid_params = {'batch_size': config.batch_size, 'shuffle': False, 'num_workers': 0, 'pin_memory': True} if use_cuda else {}

# load shapes of all LIST dataset
vol_shapes = pickle.load(open('./volume_shapes.pkl', 'rb'))
N = 131   # LIST sample size
#N = 4
img_files = ['volume-{}.nii'.format(i) for i in range(N)]       # image files
seg_files = ['segmentation-{}.nii'.format(i) for i in range(N)] # segmentation files
# select (train, test) 3D (volume) images
train_ind, test_ind, _, _ = train_test_split(np.arange(N), np.arange(N), test_size=0.25)

# split 3D images into pairs: (vol i, slice j)
train_pairs = [(i, j) for i in train_ind for j in range(vol_shapes['volume{}'.format(i)][2])]
test_pairs = [(i, j) for i in test_ind for j in range(vol_shapes['volume{}'.format(i)][2])]

# Data loader
train_loader = DataLoader(Dataset(data_path, train_pairs), **train_params)
valid_loader = DataLoader(Dataset(data_path, test_pairs), **valid_params)

# U-Net model
net = UNet(num_classes=config.num_classes).to(device)

print("Using", torch.cuda.device_count(), "GPU!")
loss_func = nn.CrossEntropyLoss(reduction='sum').cuda()
net_params = list(net.parameters())
optimizer = torch.optim.Adam(net_params, config.lr)

check_mkdir(ckpt_path)
check_mkdir(os.path.join(ckpt_path, exp_name))
path_epoch2='/mnt/ibrixfs05-Radiomics/Sunan_liver_seg/Segmentation_results/model_ckpt/LITS_UNET/train_epoch2/model_checkpoint'
checkpoint = torch.load(path_epoch2)
net.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
# train & test models
for epoch in range(2, config.epoch_num):
    # training model
    model_new, op_new,train_save_dir=train(train_loader, net, loss_func, optimizer, epoch, config, no_transform, visualize, device)
    check_p={'epoch': epoch,'model_state_dict': model_new.state_dict(),'optimizer_state_dict': op_new.state_dict()}
    torch.save(check_p, os.path.join(train_save_dir, 'model_checkpoint'))
    # validating model
  #  val_loss = validate(valid_loader, net, loss_func, optimizer, epoch, config, no_transform, visualize, device)



"""
checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
"""

