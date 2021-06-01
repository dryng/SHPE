import time
import utils
import model as m
import torch.nn as nn
import torch
import numpy as np
import visualizations

def train_single_lr(model, train_iterator, val_iterator, lr, n_epochs, device, path):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []

    best_valid_loss = float('inf')

    for epoch in range(n_epochs):
        start_time = time.time()

        train_loss, _ = train_one_epoch(model, train_iterator, optimizer, device)
        val_loss, _ = evaluate(model, val_iterator, device)

        end_time = time.time()

        # add each loss to loss lists
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        epoch_mins, epoch_secs = utils.epoch_time(start_time, end_time)

        # save model
        if val_loss < best_valid_loss:
            best_valid_loss = val_loss
            torch.save({
                'epoch': epoch,
                'lr': lr,
                'model': model,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss
            }, 'saved_models/' + path + '.pt')
            # torch.save(model.state_dict(), save_filename)

        # print stats every epoch
        print(f'Epoch: {epoch + 1:02} / {n_epochs} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'Train Loss: {train_loss:.3f}')
        print(f'Val. Loss: {val_loss:.3f}')

        # print loss curves every 10 epochs
        if (epoch + 1) % 10 == 0:
            visualizations.plot_loss_curves(train_losses, val_losses, epoch + 1, n_epochs)


'''
Train a given model for 1 epoch
Params:
    model
    iterator
    optimizer
    criterion
    device
Returns:
    loss over 1 epoch
'''
def train_one_epoch(model, iterator, optimizer, device):
    model.train()

    epoch_loss = 0
    losses = []
    for i, (x, y) in enumerate(iterator):
        if i == 0:
            print(f"Curr lr: {optimizer.state_dict()['param_groups'][0]['lr']}")
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        output = model(x)
        loss = masked_loss_for_joint_visibilty(output.float(), y.float(), device)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        losses.append(loss.item())

    return epoch_loss / len(iterator), losses


'''
Calculate loss for dataset w/o updating parameters (i.e validation loss)
Params:
    model
    iterator
    criterion
    device
Returns:
    loss over a dataset
'''
def evaluate(model, iterator, device):
    model.eval()

    losses = []
    epoch_loss = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(iterator):
            x = x.to(device)
            y = y.to(device)

            output = model(x)
            loss = masked_loss_for_joint_visibilty(output.float(), y.float(), device)

            epoch_loss += loss.item()
            losses.append(loss.item())

    return epoch_loss / len(iterator), losses


# check this
def masked_loss_for_joint_visibilty(output, y, device):
    #criterion1 = nn.BCEWithLogitsLoss() # adds sigmoid
    criterion1 = nn.BCELoss()
    criterion2 = nn.MSELoss()
    loss = 0

    # first 16 nodes -> whether joint is visible or not
    # convert to numpy on cpu to insert
    mask = y[:,0:16].cpu().numpy()

    # double mask values to cover both x and y corrdinates
    for i in range(0,mask.shape[1]*2, 2):
        mask = np.insert(mask,i+1,mask[:,i],axis=1)

    # back to torch -> cpu
    mask = torch.from_numpy(mask).to(device)

    for i in range(16):
        # first 16 if found labels
        item1 = output[:, i]
        #print(item1)
        item2 = y[:,i]
        #print(item2)
        loss += criterion1(output[:, i], y[:, i])
        # coordinates -> mask the ones are not found

    # mask joints for those that aren't visible (output and label)
    masked_output = output[:,16:] * mask
    masked_y = y[:,16:] * mask
    #print(masked_output)
    #output[:,16:] = output[:,16:] * mask
    #print(output)
    #y[:,16:] = y[:,16:] * mask
    #print(y)

    loss += criterion2(masked_output, masked_y)

    return loss
