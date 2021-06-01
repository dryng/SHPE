import os
import torch
import torchvision
import torchvision.transforms.functional as TF
import random
from torchvision import transforms
from scipy.io import loadmat
from torch.utils.data import Dataset, DataLoader
from skimage import io
from PIL import Image


# create dataset and return iterators
# need to spilt into train,val,test
# figure out how to make dataset bigger with agumentation
def load_data(image_size, batch_size, root_dir, annotation_path, joint_visibilty=False):

    train_transform = transforms.Compose([
                                    transforms.Resize((image_size, image_size)),
                                    #transforms.RandomCrop(image_size, padding=4),
                                    #transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                    ])

    test_transform = transforms.Compose([
                                transforms.Resize((image_size, image_size)),
                                transforms.ToTensor(),
                                # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                ])

    mydata = CleanData(annotation_path)
    data = mydata.get_complete_data()

    # need to figure out how to split into train and test with diff transformers
    dataset = SHPEDataSet(root_dir, data, joint_visibilty, transform=train_transform)
    train_set, val_set = torch.utils.data.random_split(dataset, [len(dataset) - 1000, 1000])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=8)

    return train_loader, val_loader

# Data set
# bottleneck is reading in image
class SHPEDataSet(Dataset):
    def __init__(self, root_dir, data, joint_visibilty, transform=None):
        self.root_dir = root_dir
        self.data = data
        self.transform = transform
        self.joint_visibilty = joint_visibilty

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        img_id = self.data[item]["image"]
        img_path = os.path.join(self.root_dir, img_id)
        image = Image.open(img_path)
        #image.save('/Users/danny/Desktop/' + 'before' + str(item), "JPEG")

        og_img_size = (image.height, image.width)

        # need flatten points
        if self.transform:
            image = self.transform(image)

        #temp_im = transforms.ToPILImage()(image.squeeze_(0))
        #temp_im.save('/Users/danny/Desktop/' + 'after' + str(item), "JPEG")

        new_img_size = (image.shape[1], image.shape[2])
        #update scales
        h_scale = new_img_size[1]/og_img_size[1]
        w_scale = new_img_size[0]/og_img_size[0]

        # take x percentage of height scale for x's (new image height / og img height) * xs
        # take x percentage of width scale for y's (new image width/ og img width) * ys
        labels = self.data[item]["flattened"]
        # pass by reference change

        self.scale_labels(labels, h_scale, w_scale)

        #add visibility to labels
        if self.joint_visibilty:
            jv = self.data[item]['visible']
            for i in range(len(jv)):
                labels.insert(i, int(jv[i]))

        labels = torch.tensor(labels)

        return (image, labels)

    '''
    Do all transforms in here
    params:
        h_scale: image height scaling
        w_scale: image width scaling
    '''
    def my_transforms(self, image, label):
        if random.random() > 0.5:
            angle = random.randint(-30, 30)
            image = TF.rotate(image, angle)
            bonding_box_coordinate = TF.rotate(label, angle)
        # more transforms ...
        return image, label

    '''
    Scale coordinates in relation to image resizing
    params:
        h_scale: image height scaling
        w_scale: image width scaling
    '''
    def scale_labels(self, labels, h_scale, w_scale):
        for i in range(len(labels)):
            # even = x
            if (i % 2) == 0:
                labels[i] = int(labels[i] * h_scale)
            # odd = y
            else:
                labels[i] = int(labels[i] * h_scale)

def scale_labels_back(labels, h_scale, w_scale):
    for i in range(len(labels)):
        # even = x
        if (i % 2) == 0:
            labels[i] = int(labels[i] / h_scale)
        # odd = y
        else:
            labels[i] = int(labels[i] / w_scale)

    return labels

'''
Custom Transform to scale coordinates in relation to image resizing
params:
    h_scale: image height scaling
    w_scale: image width scaling
    
I dont think this will work -> scales need to change every call
'''
class ScaleLabels:
    def __init__(self, h_scale, w_scale):
        self.h_scale = h_scale
        self.w_scale = w_scale

    def __call__(self, y):
       pass


'''
Reads matlab file and returns data, a list of dictionaries
Params:
    filepath : path to file containing annotations
Returns:
    data:
        [{image: "image", xs: "x2 cordinates", ys: "y cordinates"}, {...}, 
'''
# need complete annotation for training set
class CleanData:
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = self.load_data()
        self.ordered_data, self.complete_data = self.set_ordered_coordinates()
        #self.complete_examples = 0
        #self.num_valid = 0
        #self.num_total = 0

    def get_data(self):
        return self.data

    def get_ordered_data(self):
        return self.ordered_data

    # data has been ordered and flattened appended
    def get_complete_data(self):
        return self.complete_data

    # ordered cordinates for specific index
    def ordered_coordinates(self, index):
        ids = self.data[index]["ids"]
        xs = self.data[index]["xs"]
        ys = self.data[index]["ys"]
        ordered = []
        # should go from 1 -> 15
        for i in range(0, 16):
            if i in ids:
                index = ids.index(i)
                ordered.append((xs[index], ys[index]))
            else:
                ordered.append((-1,-1))
        # count fully anonated
        return ordered, (len(ids) == 16)

    # list of points (x,y) and flatten them into single list [x1, y1, x2, y2, ... ]
    # call after ordering
    def flatten_coordinates(self, points):
        flat = []
        for point in points:
            flat.append(int(point[0]))
            flat.append(int(point[1]))
        return flat

    # order all cordinates in dataset
    def set_ordered_coordinates(self):
        ordered_data = []
        complete_data = []
        self.complete_examples = 0
        for i, item in enumerate(self.data):
            ordered, full = self.ordered_coordinates(i)
            if full:
                self.complete_examples += 1
                flattened = self.flatten_coordinates(ordered)
                complete_data.append({"image": item["image"], "coordinate": ordered, "flattened": flattened, "visible": self.data[i]["visible"]})
            else:
                ordered_data.append({"image": item["image"], "coordinate": ordered})

        return ordered_data, complete_data

    def load_data(self):
        data = loadmat(self.filepath)
        data = data["RELEASE"]
        annolist = data["annolist"]
        data = []
        i = 1
        exception_count = 0
        for a in annolist:
            for b in annolist:
                for c in b:
                    for d in c:
                        for e in d:
                            #first 4 don't have points for somereason
                            if i > 4 and e[1].shape == (1,1) and e[2].size > 0:
                                d = {}
                                try:
                                    image = e[0][0]['name'][0].item()
                                    annopoints = e[1][0]['annopoints']
                                    p_layer1 = annopoints[0][0]
                                    #x points
                                    xs = [x for x in p_layer1[0][0]['x'].squeeze()]
                                    xs = [x[0][0] for x in xs]
                                    #y points
                                    ys = [y for y in p_layer1[0][0]['y'].squeeze()]
                                    ys = [y[0][0] for y in ys]
                                    #ids
                                    ids = [id for id in p_layer1[0][0]['id'].squeeze()]
                                    ids = [id[0][0] for id in ids]
                                    #join visibility
                                    vis = [vi for vi in p_layer1[0][0]['is_visible'].squeeze()]
                                    v = 0
                                    for vi in vis:
                                        # empty array -> set to 0
                                        if vi.size == 0:
                                            vis[v] = 0
                                        # if its not empty -> grab first thing
                                        else: #vi[0][0] == 1:
                                            vis[v] = vi[0][0]
                                        v += 1
                                    #vis = [vi[0][0] for vi in vis if vi[0][0] != []] # need to set to 0 if empty

                                    d["image"] = image
                                    d["xs"] = xs
                                    d["ys"] = ys
                                    d["ids"] = ids
                                    d["visible"] = vis

                                    data.append(d)
                                except Exception as e:
                                    #print(e)
                                    exception_count += 1
                                    pass
                            i += 1 

        self.length = len(data)

        return data
