'''
myData = data.CleanData("mpii_human_pose_v1_u12_1.mat")
data = myData.get_data()
ordered_data = myData.get_ordered_data()
complete_data = myData.get_complete_data()
print(f'Valid: {myData.length}')
print(f'Num complete: {myData.complete_examples}')

for i in range(len(complete_data)):
    annotate.annotate_flat_image(images_base_filepath + complete_data[i]["image"], complete_data[i]["flattened"])


annoations_filepath = "mpii_human_pose_v1_u12_1.mat"
images_base_filepath = "/Users/danny/Code/Data/SHPE/images/"

myData = data.CleanData("mpii_human_pose_v1_u12_1.mat")
data = myData.get_data()
ordered_data = myData.get_ordered_data()
complete_data = myData.get_complete_data()
print(f'Valid: {myData.length}')
print(f'Num complete: {myData.complete_examples}')

for i in range(len(complete_data)):
    annotate.annotate_flat_image(complete_data[i]["flattened"], image_path=images_base_filepath + complete_data[i]["image"])


    #example
for i in range(10):
    itr = iter(train_loader)
    x,y = itr.next()

    print(f"Shape of x: {x.shape}")
    print(f"Shape of Y: {y.shape}")
    # go over batch
    for i in range(x.shape[0]):
        # img = x[i]
        #img = x[i].permute((1, 2, 0)).numpy().copy()
        img = np.asarray(transforms.ToPILImage()(x[i].squeeze_(0)))
        coordinates = y[i]
        annotate.annotate_flat_image(coordinates,image=img)
'''