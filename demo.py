import torch
from torch.autograd import Variable
import utils
import dataset
from PIL import Image
import os
import models.crnn as crnn


def image_file_name(file_dir):   
    L=[]   
    for root, dirs, files in os.walk(file_dir):  
        for file in files:  
            if os.path.splitext(file)[1] == '.jpg' or os.path.splitext(file)[1] == '.png':  
                L.append(file)  
    return L  

model_path = 'backup/netCRNN_24_40.pth'
img_path_root = 'data/data_chetou_QR/crnn_test_img_gt'
alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'

model = crnn.CRNN(32, 1, 37, 256)
if torch.cuda.is_available():
    model = torch.nn.DataParallel(model).cuda()
print('loading pretrained model from %s' % model_path)
model.load_state_dict(torch.load(model_path))

converter = utils.strLabelConverter(alphabet)

transformer = dataset.resizeNormalize((100, 32))

imageNames=image_file_name(img_path_root)

for imageName in imageNames:
    image = Image.open(os.path.join(img_path_root,imageName)).convert('L')
    image = transformer(image)
    if torch.cuda.is_available():
        image = image.cuda()
    image = image.view(1, *image.size())
    image = Variable(image)

    model.eval()
    preds = model(image)

    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)

    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
    print('%-20s => %-20s' % (raw_pred, sim_pred))
    txtName=os.path.join(img_path_root,imageName.replace('jpg','txt'))
    with open(txtName,'w') as f:
        f.write(sim_pred)