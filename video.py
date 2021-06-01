import numpy as np
import torchvision
import torchvision.transforms.functional as TF
import cv2
import annotate
import training
import torch
import data

def read_from_video(model, im_size, device):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    while True:
        # capture fram by frame
        ret, frame = cap.read()

        # if frame is read correctly - ret is True
        if not ret:
            print("Can't recieve frame (stream end?). Exiting...")
            break
        # operations onq
        # frames here
        # frame = img -> preproccess -> call predict

        og_frame = frame

        # should do all this in predict not here
        frame = cv2.resize(frame, dsize=(im_size,im_size))
        #frame = frame.reshape(3, im_size, im_size)
        frame = TF.to_tensor(frame)
        frame = TF.normalize(frame, mean= (0.4914, 0.4822, 0.4465), std= (0.2023, 0.1994, 0.2010))

        h_scale = im_size / og_frame.shape[1]
        w_scale = im_size / og_frame.shape[0]


        coordinates = training.predict(model, frame.unsqueeze(0).to(device))
        scaled_coordinates = data.scale_labels_back(coordinates[:,16:].squeeze().tolist(), h_scale, w_scale)
        coordinates = coordinates.squeeze().tolist()[0:16] + scaled_coordinates
        frame = annotate.get_annotate_flat_image(coordinates, image=og_frame)

        '''
        coordinates = training.predict(model, frame.unsqueeze(0).to(device))
        coordinates = data.scale_labels_back(coordinates.squeeze().tolist(), h_scale, w_scale)
        frame = annotate.get_annotate_flat_image(coordinates, image=og_frame)
        '''
        # display frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # when everything done, realse capture
    cap.release()
    cv2.destroyAllWindows()