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
    criterion = nn.MSELoss()

    train_losses = []
    val_losses = []

    best_valid_loss = float('inf')

    for epoch in range(n_epochs):
        start_time = time.time()

        train_loss, _ = train_one_epoch(model, train_iterator, optimizer, criterion, device)
        val_loss, _ = evaluate(model, val_iterator, criterion, device)

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


def train_cyclic_lr(model, train_iterator, val_iterator, base_lr, max_lr, step_size, n_epochs, device, path):
    model.train()

    #optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)
    optimizer = torch.optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=5e-4)
    criterion = nn.MSELoss()
    best_valid_loss = float('inf')

    train_losses = []
    val_losses = []

    # example uses lr=.1, step=90, gamma=.1
    # Documentation for step:
    # Assuming optimizer uses lr = 0.05, step=30, gamma=.1 for all groups
    # lr = 0.05     if epoch < 30
    # lr = 0.005    if 30 <= epoch < 60
    # lr = 0.0005   if 60 <= epoch < 90
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr, max_lr, step_size, 10000, mode='triangular')

    for epoch in range(n_epochs):
        start_time = time.time()
        train_loss, _ = train_one_epoch_scheduler(model, train_iterator, optimizer, criterion, scheduler, device)
        val_loss, _ = evaluate(model, val_iterator, criterion, device)

        # add each loss to loss lists
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # scheduler.step()

        end_time = time.time()

        epoch_mins, epoch_secs = utils.epoch_time(start_time, end_time)

        # save model
        if val_loss < best_valid_loss:
            best_valid_loss = val_loss
            torch.save({
                'epoch': epoch,
                'base_lr': base_lr,
                'max_lr': max_lr,
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



def train_one_epoch_scheduler(model, iterator, optimizer, criterion, scheduler, device):
    model.train()

    epoch_loss = 0
    losses = []
    #start = time.time()
    for i, (x, y) in enumerate(iterator):
        #end = time.time()
        #mins, secs = helpers.epoch_time(start, end)
        #print(f'M: {mins}, S: {secs}')
        #if i == 0:
        # print for all to make sure working
        # print(f"Curr lr: {optimizer.state_dict()['param_groups'][0]['lr']}")
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        output = model(x)
        loss = criterion(output.float(), y.float())

        loss.backward()
        optimizer.step()

        # scheduler
        scheduler.step()

        epoch_loss += loss.item()
        losses.append(loss.item())

        start = time.time()

    return epoch_loss / len(iterator), losses



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
def train_one_epoch(model, iterator, optimizer, criterion, device):
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
        loss = criterion(output.float(), y.float())

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
def evaluate(model, iterator, criterion, device):
    model.eval()

    losses = []
    output_coordinates = []
    labels = []
    epoch_loss = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(iterator):
            x = x.to(device)
            y = y.to(device)

            output = model(x)
            loss = criterion(output.float(), y.float())
            output_coordinates.append(output)
            labels.append(y)

            epoch_loss += loss.item()
            losses.append(loss.item())

    return epoch_loss / len(iterator), losses


def predict(model, x):
    model.eval()
    return model(x)

'''
See if model can overfit on N training examples
N being small (i.e) 20
This is a santity check to make sure model can converge
Params:
    model
    iterator
    optimizer
    criterion
    stop: number of examples to train on
    device
Returns:
    loss over stop examples
'''
def check_overfit(model, iterator, optimizer, criterion, n_epochs, stop, device):
    model.train()

    for epoch in range(n_epochs):
        epoch_loss = 0
        #start = time.time()
        for i, (x, y) in enumerate(iterator):
            #end = time.time()
            #mins, secs = utils.epoch_time(start,end)
            #print(f'M: {mins}, S: {secs}')

            if i == stop:
                break
            else:
                x = x.to(device)
                y = y.to(device)

                optimizer.zero_grad()

                # output = 10
                # y = 32
                output = model(x)
                loss = criterion(output.float(), y.float())

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            #start = time.time()

        print(f"Epoch: {epoch}/{n_epochs}. Loss: {epoch_loss / stop}")


'''
Explore to see what lr works the best for this model and dataset
Params:
    lr low: low range for random (ex 1e-2)
    lr high: high range for random (ex 1e-5)
    max count: number of different lr to try
    n_epochs: num epochs for each lr
    model_size
    device
Returns:

'''
def corse(small_model_size, large_model_size, lr_low, lr_high, max_count, n_epochs, train_loader, test_loader, device):
    for i in range(max_count):
        m_size = 0
        lr = np.random.uniform(lr_low, lr_high)
        if i < max_count / 2:
            model = m.get_model_resnet(small_model_size, device)
            m_size = small_model_size
        else:
            model = m.get_model_resnet(large_model_size, device)
            m_size = large_model_size
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        start_time = time.time()

        for epoch in range(n_epochs):
            train_loss, _ = train_one_epoch(model, train_loader, optimizer, criterion, device)
            valid_loss, _ = evaluate(model, test_loader, criterion, device)

        end_time = time.time()

        epoch_mins, epoch_secs = utils.epoch_time(start_time, end_time)

        with open(f'tuning/corse.txt', 'a') as fp:
            fp.write(
                f'({i}/{max_count}), Model Size: {m_size}, Train Loss: {train_loss:.3f}, Val. Loss: {valid_loss:.3f}, lr: {lr}, Mins: {epoch_mins}, Secs: {epoch_secs}) \n')

        print(
            f'({i}/{max_count}), Model Size: {m_size}, Train Loss: {train_loss:.3f}, Val. Loss: {valid_loss:.3f}, lr: {lr}, Mins: {epoch_mins}, Secs: {epoch_secs}')
