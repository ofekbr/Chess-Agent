import math

import chess.pgn
import numpy as np
from matplotlib import transforms
from torch import nn, optim
from torch.utils import data
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from chess import engine
from timeit import default_timer as timer
from datetime import timedelta


class NeuralNetwork(nn.Module):
    def __init__(self):
        """
        Declare layers for the model
        """
        super().__init__()
        self.fc1 = nn.Linear(6, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 2)
        self.norm1 = nn.BatchNorm1d(128)
        self.norm2 = nn.BatchNorm1d(32)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.norm2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x


def nnTrain(threats_path, moves_available_path, clock_path, taken_path, move_num_path, materials_path, times_path):
    torch.manual_seed(0)

    input_threats = np.loadtxt(threats_path, dtype=int).tolist()
    input_moves_available = np.loadtxt(moves_available_path, dtype=int).tolist()
    input_clock = np.loadtxt(clock_path, dtype=int).tolist()
    input_taken = np.loadtxt(taken_path).tolist()
    input_move_num = np.loadtxt(move_num_path, dtype=int).tolist()
    input_materials = np.loadtxt(materials_path, dtype=int).tolist()
    input_times = np.loadtxt(times_path, dtype=int)
    temp_input_times = np.zeros((len(input_times)), dtype=int)
    mean = np.mean(input_times)
    std = np.std(input_times)
    top_first_class = 2
    # top_second_class = 10
    # top_third_class = 35
    zeros = 0
    ones = 0
    for i in range(len(input_times)):
        if input_times[i] <= top_first_class:
            temp_input_times[i] = 0
            zeros += 1
        # elif input_times[i] <= top_second_class:
        #     temp_input_times[i] = 1
        # elif input_times[i] <= top_third_class:
        #     temp_input_times[i] = 2
        else:
            temp_input_times[i] = 1
            ones += 1
    print(f'zeros = {zeros}    ones = {ones}')
    print(f'zeros = {zeros/len(input_threats)}    ones = {ones/len(input_threats)}')
    input_times = temp_input_times
    print(len(input_threats))
    print(len(input_moves_available))
    print(len(input_clock))
    print(len(input_taken))
    print(len(input_move_num))
    print(len(input_materials))
    print(len(input_times))

    threat_array = np.array(input_threats)
    moves_available_array = np.array(input_moves_available)
    clock_array = np.array(input_clock)
    move_num_array = np.array(input_move_num)
    taken_array = np.array(input_taken)
    input_materials = np.array(input_materials)
    input_times = np.array(input_times)
    complete_array = np.vstack(
        (threat_array, moves_available_array, clock_array, taken_array, move_num_array, input_materials)).transpose()
    complete_array = complete_array.astype(float)

    tensor_games = torch.Tensor(complete_array)
    tensor_times = torch.Tensor(input_times)

    my_dataset = data.TensorDataset(tensor_games, tensor_times)

    model = NeuralNetwork()
    training_data, validation_data = data.random_split(my_dataset, [21000, 5058])
    train_loader = data.DataLoader(training_data, batch_size=64, shuffle=True)
    val_loader = data.DataLoader(validation_data, batch_size=128, shuffle=True)
    learning_rate = 0.0005
    epochs = 10
    optimizer = optim.Adam(model.parameters(), learning_rate)
    weight = torch.tensor([5.0, 1.0])
    criterion = nn.NLLLoss(weight=weight)  # weight=weight

    # sum = 0
    # for i in training_data:
    #     sum += int(i[1])
    # avg = sum / len(training_data)

    def train_model(model, optimizer, criterion, epochs, train_loader, val_loader):
        train_losses, val_losses, avg_losses = [], [], []
        for e in range(epochs):
            running_loss = 0
            running_val_loss = 0
            for images, labels in train_loader:
                # Training pass
                model.train()
                optimizer.zero_grad()
                output = model.forward(images)
                labels = torch.squeeze(labels)
                labels = labels.type(torch.LongTensor)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
            else:
                val_loss = 0
                # 6.2 Evalaute model on validation at the end of each epoch.
                accuracy = 0
                close = 0
                with torch.no_grad():
                    for images, labels in val_loader:
                        output = model.forward(images)
                        pred = output.max(1, keepdim=True)[1]
                        accuracy += pred.eq(labels.view_as(pred)).sum().item()

                        labels = torch.squeeze(labels)
                        labels = labels.type(torch.LongTensor)

                        if e == epochs - 1:
                            for image, i, j in zip(images, output, labels):
                                print(f"Our guess: {str(int(np.argmax(i)))} Label: {str(int(j))}")
                                if int(np.argmax(i)) == int(j):
                                    close += 1
                                # print(f"Features: {str(image)} Our guess: {str(int(np.argmax(i)))} Label: {str(int(j))}")
                                # if int(np.argmax(i)) == 1 and int(j)==0:
                                #     guess_one_label_zero+=1
                                # elif int(np.argmax(i)) == 0 and int(j)==1:
                                #     guess_zero_label_one+=1
                        val_loss = criterion(output, labels)
                        running_val_loss += val_loss.item()

                # 7. track train loss and validation loss
            train_losses.append(running_loss / len(train_loader))
            val_losses.append(running_val_loss / len(val_loader))

            print("Epoch: {}/{}.. ".format(e + 1, epochs),
                  "Training Loss: {:.3f}.. ".format(running_loss / len(train_loader.dataset)),
                  "Validation Loss: {:.3f}.. ".format(running_val_loss / len(val_loader.dataset)),
                  "Accuracy: {:.3f}.. ".format(100 * (accuracy / len(val_loader.dataset))))
            print(close / len(val_loader.dataset))
        return train_losses, val_losses

    train_losses, val_losses = train_model(model, optimizer, criterion, epochs, train_loader, val_loader)
    list = []
    for i in range(10):
        list.append(i)
    plt.plot(list, train_losses)
    plt.plot(list, val_losses)
    plt.show()


nnTrain('new_threats.txt', 'new_available_moves.txt', 'new_clock.txt', 'new_taken.txt', 'new_count_moves.txt',
        'new_materials.txt', 'new_times.txt')
