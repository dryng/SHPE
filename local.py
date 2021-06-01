import torch
import torch.nn as nn
import data
import annotate
import numpy as np
import model as m
import training
import training_v2
import os
import video
from skimage import io
from torchvision import transforms
from PIL import Image
from torchvision import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)
print(torch.__version__)

annoations_filepath = "mpii_human_pose_v1_u12_1.mat"
images_base_filepath_local = "/Users/danny/Code/Data/SHPE/images/"

#train_loader, val_loader = data.load_data(227, 128, images_base_filepath_local, annoations_filepath)

#model_ft = models.resnet34(pretrained=False)
#num_ftrs = model_ft.fc.in_features
#model_ft.fc = nn.Linear(num_ftrs, 32)
#model = model_ft.to(device)

#model = m.get_model_resnet(34, device)
#model = m.get_model_resnet_no_vis(34, device)
#train_loader, val_loader = data.load_data(227,128,"/Users/danny/Code/Data/SHPE/images/",annoations_filepath,joint_visibilty=True)

# load model and test it
checkpoint = torch.load('saved_models/visibility_with_sigmoid.pt', map_location=device)
model = checkpoint['model']
model.load_state_dict(checkpoint['model_state_dict'])
print(f"Epoch: {checkpoint['epoch']}")
print(f"Train loss from checkpoint: {checkpoint['train_loss']}")
print(f"Val loss from checkpoint: {checkpoint['val_loss']}")
#loss, _= training_v2.evaluate(model, val_loader, device)
#print(f"loss after eval: {loss} ")

#live video
video.read_from_video(model, 227, device)

'''
# test images 
itr = iter(train_loader)
(x, y), (h, w), img_paths = itr.next()
for i in range(x.shape[0]):
    img_path = os.path.join(images_base_filepath_local, img_paths[i])
    #img = np.asarray(transforms.ToPILImage()(x[i].squeeze_(0)))
    coordinates = training.predict(model, x[i].unsqueeze(0).to(device))
    coordinates = data.scale_labels_back(coordinates.squeeze().tolist(), h[i], w[i])
    annotate.annotate_flat_image(coordinates, image_path=img_path)

    
    img = np.asarray(transforms.ToPILImage()(x[i].squeeze_(0)))
    coordinates = training.predict(model, x[i].unsqueeze(0).to(device))
    annotate.annotate_flat_image(coordinates.squeeze(), image=img)
    
'''
