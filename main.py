import torch
import torch.nn as nn
import data
#import annotate
import numpy as np
import model as m
import training
import training_v2
from skimage import io
#import cv2
import matplotlib.pyplot as plt
from torchvision import transforms, models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)
print(torch.__version__)

annoations_filepath = "mpii_human_pose_v1_u12_1.mat"
images_base_filepath_local = "/Users/danny/Code/Data/SHPE/images/"
images_base_filepath_server = "/home/paperspace/Code/Data/SHPE/images/"

train_loader, val_loader = data.load_data(227,128,images_base_filepath_server,annoations_filepath, True)


# transfer learning: fine tuning
'''
model_ft = models.resnet34(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 32)
model = model_ft.to(device)
'''

model = m.get_model_resnet(34, device)
print(model)
#m.observe_shape(model, 227, device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()


#training.corse(18, 34, 1e-1, 1e-4, 100, 5, train_loader, val_loader, device)

#training.check_overfit(model, train_loader, criterion=criterion, device=device, n_epochs=50, optimizer=optimizer, stop=20)

#training.train_cyclic_lr(model, train_loader, val_loader, .0001, .1, 10000, 100, device, "cyclical")

#training.train_single_lr(model, train_loader, val_loader, 0.01, 75, device, 'transfer-lr-0.01')
training_v2.train_single_lr(model, train_loader, val_loader, 0.001, 75, device, 'visibility_with_sigmoid_34_norm')


loss, _, o, l= training.evaluate(model, val_loader, nn.MSELoss(), device)


itr = iter(val_loader)
x, y = itr.next()
for i in range(x.shape[0]):
    img = x[i]
    lab = y[i]
    print(lab)
    img = np.asarray(transforms.ToPILImage()(x[i].squeeze_(0)))
    #coordinates = training.predict(model, x[i].unsqueeze(0))
    #print(coordinates)

''' 
#load model and test it
checkpoint = torch.load('saved_models/cyclical.pt', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
print(f"Train loss from checkpoint: {checkpoint['train_loss']}")
print(f"Val loss from checkpoint: {checkpoint['val_loss']}")

loss, _ = training.evaluate(model, val_loader, nn.MSELoss(), device)
print(f'Val loss after loading: {loss}')
 
itr = iter(val_loader)
x, y = itr.next()
for i in range(x.shape[0]):
    img = np.asarray(transforms.ToPILImage()(x[i].squeeze_(0)))
    coordinates = training.predict(model, x[i].unsqueeze(0).to(device))

    annotate.annotate_flat_image(coordinates.squeeze(), image=img)
'''





