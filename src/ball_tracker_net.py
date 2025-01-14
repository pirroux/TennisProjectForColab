import torch
import torch.nn as nn
import numpy as np
import time
import cv2
import matplotlib.pyplot as plt
from torch.optim import Adadelta
from torch.optim.lr_scheduler import ReduceLROnPlateau
from datasets import TrackNetDataset, get_dataloaders
from trainer import plot_graph
import torch.nn.functional as F

# Define ConvBlock
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pad, bias=True, bn=True):
        super(ConvBlock, self).__init__()

        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=pad, bias=bias),
            nn.ReLU()
        ]

        if bn:
            layers.append(nn.BatchNorm2d(out_channels))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


# Define BallTrackerNet model
class BallTrackerNet(nn.Module):
    def __init__(self, out_channels=2):
        super(BallTrackerNet, self).__init__()
        self.out_channels = out_channels

        # Encoder blocks
        self.encoder = nn.ModuleList([
            ConvBlock(9, 64, 3, 1),  # First block takes 9 channels (3 frames x 3 channels)
            ConvBlock(64, 64, 3, 1),
            nn.MaxPool2d(2, 2),
            ConvBlock(64, 128, 3, 1),
            ConvBlock(128, 128, 3, 1),
            nn.MaxPool2d(2, 2),
            ConvBlock(128, 256, 3, 1),
            ConvBlock(256, 256, 3, 1),
            ConvBlock(256, 256, 3, 1),
            nn.MaxPool2d(2, 2),
            ConvBlock(256, 512, 3, 1),
            ConvBlock(512, 512, 3, 1),
            ConvBlock(512, 512, 3, 1)
        ])

        # Decoder blocks
        self.decoder = nn.ModuleList([
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ConvBlock(512, 256, 3, 1),
            ConvBlock(256, 256, 3, 1),
            ConvBlock(256, 256, 3, 1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ConvBlock(256, 128, 3, 1),
            ConvBlock(128, 128, 3, 1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ConvBlock(128, 64, 3, 1),
            ConvBlock(64, 64, 3, 1),
            ConvBlock(64, out_channels, 3, 1, bn=False)
        ])

    def forward(self, x):
        """
        Forward pass of the network
        Input: tensor of shape (batch_size, 9, height, width)
        Returns: tensor of shape (batch_size, out_channels, height, width)
        """
        try:
            # Encoder path
            for layer in self.encoder:
                x = layer(x)

            # Decoder path
            for layer in self.decoder:
                x = layer(x)

            return x

        except Exception as e:
            print(f"Error in BallTrackerNet forward pass: {str(e)}")
            return None


# Accuracy Calculation
def accuracy(y_pred, y_true):
    """
    Calculate accuracy of prediction.
    """
    correct = (y_pred == y_true).sum()
    acc = correct / (len(y_pred[0]) * y_pred.shape[0]) * 100

    non_zero = (y_true > 0).sum()
    non_zero_correct = (y_pred[y_true > 0] == y_true[y_true > 0]).sum()

    non_zero_acc = 100.0 if non_zero == 0 and non_zero_correct == 0 else 0.0 if non_zero == 0 else non_zero_correct / non_zero * 100

    return acc, non_zero_acc, non_zero_correct


# Display Results
def show_result(inputs, labels, outputs):
    """
    Display network’s prediction results.
    """
    num_classes = outputs.size(1)
    outputs = outputs.argmax(dim=1).detach().cpu().numpy()
    if num_classes == 2:
        outputs *= 255

    mask = outputs[0].reshape((360, 640))
    fig, ax = plt.subplots(1, 2, figsize=(20, 1 * 5))
    ax[0].imshow(inputs[0, :3, :, ].detach().cpu().numpy().transpose((1, 2, 0)))
    ax[0].set_title('Image')
    ax[1].imshow(labels[0].detach().cpu().numpy().reshape((360, 640)), cmap='gray')
    ax[1].set_title('gt')
    plt.show()
    plt.figure()
    plt.imshow(mask, cmap='gray')
    plt.title('Pred')
    plt.show()


# Distance Calculation
def get_center_ball_dist(output, x_true, y_true, num_classes=256):
    """
    Calculate distance of predicted center from the real center.
    Success if distance is less than 5 pixels, fail otherwise.
    """
    max_dist = 5
    success, fail = 0, 0
    dists = []
    Rx = 640 / 1280
    Ry = 360 / 720

    for i in range(len(x_true)):
        x, y = -1, -1
        cur_output = output[i].reshape((360, 640))

        cur_output = cur_output.astype(np.uint8)
        heatmap = cv2.resize(cur_output, (640, 360))

        if num_classes == 256:
            ret, heatmap = cv2.threshold(heatmap, 127, 255, cv2.THRESH_BINARY)
        else:
            heatmap *= 255

        circles = cv2.HoughCircles(heatmap, cv2.HOUGH_GRADIENT, dp=1, minDist=1, param1=50, param2=2, minRadius=2, maxRadius=7)

        if circles is not None:
            if len(circles) == 1:
                x = int(circles[0][0][0])
                y = int(circles[0][0][1])

        if x_true[i] < 0:
            if x < 0:
                success += 1
            else:
                fail += 1
            dists.append(-2)
        else:
            if x < 0:
                fail += 1
                dists.append(-1)
            else:
                dist = np.linalg.norm(((x_true[i] * Rx) - x, (y_true[i] * Ry) - y))
                dists.append(dist)
                if dist < max_dist:
                    success += 1
                else:
                    fail += 1

    success_rate = success / len(x_true) * 100
    fail_rate = fail / len(x_true) * 100

    return dists, success_rate, fail_rate


# def train(model_saved_state=None, epochs_num=100, lr=1.0, num_classes=256, batch_size=1):
#     """
#     Training TrackNet model
#     :param model_saved_state: saved state of the model
#     :param epochs_num: number of epochs to train
#     :param lr: learning rate
#     :param num_classes: number of output classes of the model
#     :param batch_size: batch size to use for training
#     """

#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     print(f'Using device {device}')
#     model = BallTrackerNet(out_channels=num_classes, bn=True)
#     train_losses = []
#     valid_losses = []
#     train_acc = []
#     valid_acc = []
#     train_success_epochs = []
#     train_fail_epochs = []
#     valid_success_epochs = []
#     valid_fail_epochs = []
#     total_epochs = 0
#     # Load model if saved state was given
#     if model_saved_state is not None:
#         saved_state = torch.load(model_saved_state)
#         model.load_state_dict(saved_state['model_state'])
#         train_losses = saved_state['train_loss']
#         valid_losses = saved_state['valid_loss']
#         train_acc = saved_state['train_acc']
#         valid_acc = saved_state['valid_acc']
#         total_epochs = saved_state['epochs']
#         train_success_epochs = saved_state['train_success']
#         train_fail_epochs = saved_state['train_fail']
#         valid_success_epochs = saved_state['valid_success']
#         valid_fail_epochs = saved_state['valid_fail']
#         print('Loaded saved state')
#     model.to(device)

#     # DataLoaders
#     train_dl, valid_dl = get_dataloaders('../dataset/Dataset/training_model2.csv', root_dir=None, transform=None,
#                                          batch_size=batch_size, num_classes=num_classes, dataset_type='tracknet',
#                                          num_workers=2)
#     # Loss function
#     criterion = nn.CrossEntropyLoss()
#     # Optimizer
#     optimizer = Adadelta(model.parameters(), lr=lr)
#     # Scheduler
#     lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8, verbose=True,
#                                      min_lr=0.000001)

#     for epoch in range(epochs_num):
#         start_time = time.time()
#         for phase in ['train', 'val']:
#             if phase == 'train':
#                 model.train(True)  # Set model to training mode
#                 dl = train_dl
#                 steps_per_epoch = 400 / batch_size

#             else:
#                 model.train(False)  # Set model to evaluate mode
#                 dl = valid_dl
#                 steps_per_epoch = 200 / batch_size
#             print(f'Starting Epoch {epoch + 1} Phase {phase}')
#             running_loss = 0.0
#             running_acc = 0.0
#             running_no_zero_acc = 0.0
#             running_no_zero = 0
#             min_dist = np.inf
#             running_dist = 0.0
#             count = 1
#             n1 = 0
#             n2 = 0
#             total_success = 0
#             total_fail = 0
#             for i, data in enumerate(dl):
#                 # Clear GPU cache to save space for model and input in the GPU
#                 torch.cuda.empty_cache()
#                 '''print(f'AllocMem (Mb): '
#                       f'{torch.cuda.memory_allocated() / 1024 / 1024}')'''
#                 # get the inputs; data is a list of [inputs, labels]
#                 inputs, labels = data['frames'], data['gt']
#                 inputs = inputs.to(device)

#                 labels = labels.to(device)

#                 x_true = data['x_true']
#                 y_true = data['y_true']

#                 # zero the parameter gradients
#                 optimizer.zero_grad()

#                 # forward + backward + optimize
#                 if phase == 'train':

#                     outputs = model(inputs)
#                     loss = criterion(outputs, labels)

#                     loss.backward()

#                     optimizer.step()

#                 else:

#                     with torch.no_grad():
#                         outputs = model(inputs)
#                         loss = criterion(outputs, labels)

#                 # print statistics
#                 running_loss += loss.item() * batch_size

#                 # Accuracy
#                 acc, non_zero_acc, non_zero = accuracy(outputs.argmax(dim=1).detach().cpu().numpy(),
#                                                        labels.cpu().numpy())
#                 # Prediction dist from real
#                 dists, success, fail = get_center_ball_dist(outputs.argmax(dim=1).detach().cpu().numpy(), x_true,
#                                                             y_true,
#                                                             num_classes=num_classes)
#                 total_success += success
#                 total_fail += fail
#                 for j, dist in enumerate(dists.copy()):
#                     if dist in [-1, -2]:
#                         if dist == -1:
#                             n1 += 1
#                         else:
#                             n2 += 1
#                         dists[j] = np.inf
#                     else:
#                         running_dist += dist
#                         count += 1

#                 min_dist = min(*dists, min_dist)
#                 running_acc += acc
#                 running_no_zero_acc += non_zero_acc
#                 running_no_zero += non_zero

#                 # Display results mid training
#                 if (i + 1) % 100 == 0:
#                     print('Phase {} Epoch {} Step {} Loss: {:.8f} Acc: {:.4f}%  Non zero acc: {:.4f}%  '
#                           'Success acc: {:.2f}% Min Dist: {:.4f} Avg Dist {:.4f}'.format(phase, epoch + 1, i + 1,
#                                                                                         running_loss / ((
#                                                                                                                     i + 1) * batch_size),
#                                                                                         running_acc / (i + 1),
#                                                                                         running_no_zero_acc / (i + 1),
#                                                                                         total_success * 100 / (
#                                                                                                     total_success + total_fail),
#                                                                                         min_dist, running_dist / count))
#                     print(f'n1 = {n1}  n2 = {n2}')
#                 if (i + 1) == steps_per_epoch:
#                     if phase == 'train':
#                         train_losses.append(running_loss / (i + 1))
#                         train_acc.append(running_no_zero_acc / (i + 1))
#                         train_success_epochs.append(total_success)
#                         train_fail_epochs.append(total_fail)
#                     else:
#                         valid_losses.append(running_loss / (i + 1))
#                         valid_acc.append(running_no_zero_acc / (i + 1))
#                         valid_success_epochs.append(total_success)
#                         valid_fail_epochs.append(total_fail)
#                         # lr_scheduler.step(valid_losses[-1])
#                     break

#         total_epochs += 1
#         print('Last Epoch time : {:.4f} min'.format((time.time() - start_time) / 60))
#         # Display inference mid training and saving model
#         if epoch % 50 == 49:
#             inputs, labels = data['frames'], data['gt']
#             inputs = inputs.to(device)
#             labels = labels.to(device)
#             with torch.no_grad():
#                 outputs = model(inputs)
#                 show_result(inputs, labels, outputs)

#             PATH = f'saved states/tracknet_weights_lr_{lr}_epochs_{total_epochs}.pth'
#             saved_state = dict(model_state=model.state_dict(), train_loss=train_losses, train_acc=train_acc,
#                                valid_loss=valid_losses, valid_acc=valid_acc, epochs=total_epochs,
#                                train_success=train_success_epochs, train_fail=train_fail_epochs,
#                                valid_success=valid_success_epochs, valid_fail=valid_fail_epochs)
#             torch.save(saved_state, PATH)
#             print(f'*** Saved checkpoint ***')
#     # Saving model`s weights at the end of training
#     PATH = f'saved states/tracknet_weights_lr_{lr}_epochs_{total_epochs}.pth'
#     saved_state = dict(model_state=model.state_dict(), train_loss=train_losses, train_acc=train_acc,
#                        valid_loss=valid_losses, valid_acc=valid_acc, epochs=total_epochs,
#                        train_success=train_success_epochs, train_fail=train_fail_epochs,
#                        valid_success=valid_success_epochs, valid_fail=valid_fail_epochs)
#     torch.save(saved_state, PATH)
#     print(f'*** Saved checkpoint ***')
#     print('Finished Training')
#     # Plot training results
#     plot_graph(train_losses, valid_losses, 'loss', f'../report/tracknet_losses_{total_epochs}_epochs.png')
#     plot_graph(train_acc, valid_acc, 'acc', f'../report/tracknet_acc_{total_epochs}_epochs.png')
#     plot_graph(
#         np.array(train_success_epochs) * 100 / (np.array(train_success_epochs) + np.array(train_fail_epochs)),
#         np.array(valid_success_epochs) * 100 / (np.array(valid_success_epochs) + np.array(valid_fail_epochs)),
#         'success acc', f'../report/tracknet_success_acc_{total_epochs}_epochs.png')

def train(model_saved_state=None, epochs_num=100, lr=1.0, num_classes=256, batch_size=1):
    """
    Training TrackNet model
    :param model_saved_state: saved state of the model
    :param epochs_num: number of epochs to train
    :param lr: learning rate
    :param num_classes: number of output classes of the model
    :param batch_size: batch size to use for training
    """

    # Set the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Using device {device}')

    # Initialize the model
    model = BallTrackerNet(out_channels=num_classes, bn=True)

    # Initialize lists for storing results
    train_losses = []
    valid_losses = []
    train_acc = []
    valid_acc = []
    train_success_epochs = []
    train_fail_epochs = []
    valid_success_epochs = []
    valid_fail_epochs = []
    total_epochs = 0

    # Load model if saved state was given
    if model_saved_state is not None:
        saved_state = torch.load(model_saved_state)
        model.load_state_dict(saved_state['model_state'])
        train_losses = saved_state['train_loss']
        valid_losses = saved_state['valid_loss']
        train_acc = saved_state['train_acc']
        valid_acc = saved_state['valid_acc']
        total_epochs = saved_state['epochs']
        train_success_epochs = saved_state['train_success']
        train_fail_epochs = saved_state['train_fail']
        valid_success_epochs = saved_state['valid_success']
        valid_fail_epochs = saved_state['valid_fail']
        print('Loaded saved state')

    # Move model to device
    model.to(device)

    # DataLoaders
    train_dl, valid_dl = get_dataloaders('../dataset/Dataset/training_model2.csv', root_dir=None, transform=None,
                                         batch_size=batch_size, num_classes=num_classes, dataset_type='tracknet',
                                         num_workers=2)

    # Loss function and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = Adadelta(model.parameters(), lr=lr)

    # Learning rate scheduler
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8, verbose=True,
                                     min_lr=0.000001)

    for epoch in range(epochs_num):
        start_time = time.time()
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train(True)  # Set model to training mode
                dl = train_dl
                steps_per_epoch = 400 / batch_size
            else:
                model.train(False)  # Set model to evaluate mode
                dl = valid_dl
                steps_per_epoch = 200 / batch_size

            print(f'Starting Epoch {epoch + 1} Phase {phase}')

            running_loss = 0.0
            running_acc = 0.0
            running_no_zero_acc = 0.0
            running_no_zero = 0
            min_dist = np.inf
            running_dist = 0.0
            count = 1
            n1 = 0
            n2 = 0
            total_success = 0
            total_fail = 0

            for i, data in enumerate(dl):
                # Clear GPU cache to save space for model and input in the GPU
                torch.cuda.empty_cache()

                # Get the inputs; data is a list of [inputs, labels]
                inputs, labels = data['frames'], data['gt']
                inputs = inputs.to(device)
                labels = labels.to(device)

                x_true = data['x_true']
                y_true = data['y_true']

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass, backward pass, and optimize
                if phase == 'train':
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                else:
                    with torch.no_grad():
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                # Accumulate statistics
                running_loss += loss.item() * batch_size
                acc, non_zero_acc, non_zero = accuracy(outputs.argmax(dim=1).detach().cpu().numpy(),
                                                       labels.cpu().numpy())
                dists, success, fail = get_center_ball_dist(outputs.argmax(dim=1).detach().cpu().numpy(), x_true,
                                                            y_true,
                                                            num_classes=num_classes)
                total_success += success
                total_fail += fail
                for j, dist in enumerate(dists.copy()):
                    if dist in [-1, -2]:
                        if dist == -1:
                            n1 += 1
                        else:
                            n2 += 1
                        dists[j] = np.inf
                    else:
                        running_dist += dist
                        count += 1

                min_dist = min(*dists, min_dist)
                running_acc += acc
                running_no_zero_acc += non_zero_acc
                running_no_zero += non_zero

                # Display results mid training
                if (i + 1) % 100 == 0:
                    print(f'Phase {phase} Epoch {epoch + 1} Step {i + 1} Loss: {running_loss / ((i + 1) * batch_size):.8f} '
                          f'Acc: {running_acc / (i + 1):.4f}%  Non zero acc: {running_no_zero_acc / (i + 1):.4f}%  '
                          f'Success acc: {total_success * 100 / (total_success + total_fail):.2f}% Min Dist: {min_dist:.4f} '
                          f'Avg Dist: {running_dist / count:.4f}')
                    print(f'n1 = {n1}  n2 = {n2}')

                if (i + 1) == steps_per_epoch:
                    if phase == 'train':
                        train_losses.append(running_loss / (i + 1))
                        train_acc.append(running_no_zero_acc / (i + 1))
                        train_success_epochs.append(total_success)
                        train_fail_epochs.append(total_fail)
                    else:
                        valid_losses.append(running_loss / (i + 1))
                        valid_acc.append(running_no_zero_acc / (i + 1))
                        valid_success_epochs.append(total_success)
                        valid_fail_epochs.append(total_fail)
                    break

        total_epochs += 1
        print(f'Last Epoch time : {(time.time() - start_time) / 60:.4f} min')

        # Mid-training model saving and inference
        if epoch % 50 == 49:
            inputs, labels = data['frames'], data['gt']
            inputs = inputs.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                outputs = model(inputs)
                show_result(inputs, labels, outputs)

            PATH = f'saved states/tracknet_weights_lr_{lr}_epochs_{total_epochs}.pth'
            saved_state = dict(model_state=model.state_dict(), train_loss=train_losses, train_acc=train_acc,
                               valid_loss=valid_losses, valid_acc=valid_acc, epochs=total_epochs,
                               train_success=train_success_epochs, train_fail=train_fail_epochs,
                               valid_success=valid_success_epochs, valid_fail=valid_fail_epochs)
            torch.save(saved_state, PATH)
            print(f'*** Saved checkpoint ***')

    # Final model saving
    PATH = f'saved states/tracknet_weights_lr_{lr}_epochs_{total_epochs}.pth'
    saved_state = dict(model_state=model.state_dict(), train_loss=train_losses, train_acc=train_acc,
                       valid_loss=valid_losses, valid_acc=valid_acc, epochs=total_epochs,
                       train_success=train_success_epochs, train_fail=train_fail_epochs,
                       valid_success=valid_success_epochs, valid_fail=valid_fail_epochs)
    torch.save(saved_state, PATH)
    print(f'*** Saved checkpoint ***')
    print('Finished Training')

    # Plot training results
    plot_graph(train_losses, valid_losses, 'loss', f'../report/tracknet_losses_{total_epochs}_epochs.png')
    plot_graph(train_acc, valid_acc, 'acc', f'../report/tracknet_acc_{total_epochs}_epochs.png')
    plot_graph(
        np.array(train_success_epochs) * 100 / (np.array(train_success_epochs) + np.array(train_fail_epochs)),
        np.array(valid_success_epochs) * 100 / (np.array(valid_success_epochs) + np.array(valid_fail_epochs)),
        'success acc', f'../report/tracknet_success_acc_{total_epochs}_epochs.png')


if __name__ == "__main__":
    '''state = torch.load("saved states/tracknet_weights_lr_1.0_epochs_125_256_classes.pth")
    plot_graph(state['train_loss'][5:], state['valid_loss'][5:], 'loss', '../report/tracknet_losses_150_epochs_256.png')
    plot_graph(state['train_acc'], state['valid_acc'], 'acc', '../report/tracknet_acc_150_epochs_256.png')
    plot_graph(np.array(state['train_success']) * 100 / (np.array(state['train_success']) + np.array(state['train_fail'])),
               np.array(state['valid_success']) * 100 / (np.array(state['valid_success']) + np.array(state['valid_fail'])), 'success acc', '../report/tracknet_success_acc_150_epochs_256.png')
    '''
    start = time.time()
    for lr in [0.05]:
        s = time.time()
        print(f'Start training with LR = {lr}')
        train("saved states/tracknet_weights_lr_1.0_epochs_250.pth",epochs_num=300, lr=lr, num_classes=2,
              batch_size=2)
        print(f'End training with LR = {lr}, Time = {time.time() - s}')
    print(f'Finished all runs, Time = {time.time() - start}')
