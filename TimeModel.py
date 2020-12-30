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
        self.fc3 = nn.Linear(32, 1)
        self.norm1 = nn.BatchNorm1d(128)
        self.norm2 = nn.BatchNorm1d(32)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm1(x)
        x = F.tanh(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.norm2(x)
        x = F.tanh(x)
        x = self.dropout(x)
        x = self.fc3(x)
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

    print(len(input_threats))
    print(len(input_moves_available))
    print(len(input_clock))
    print(len(input_taken))
    print(len(input_move_num))
    print(len(input_materials))
    print(len(input_times))

    input_threats = input_threats[:-3]
    input_moves_available = input_moves_available[:-3]
    input_clock = input_clock[:-3]
    input_times = input_times[:-3]
    input_move_num = input_move_num[:-3]
    input_materials = input_materials[:-3]
    input_taken = input_taken[:-3]

    threat_array = np.array(input_threats)
    moves_available_array = np.array(input_moves_available)
    clock_array = np.array(input_clock)
    move_num_array = np.array(input_move_num)
    taken_array = np.array(input_taken)
    input_materials = np.array(input_materials)
    complete_array = np.vstack(
        (threat_array, moves_available_array, clock_array, taken_array, move_num_array, input_materials)).transpose()
    # complete_array = np.insert(complete_array, 0, values=1, axis=1) # Insert values before column 1
    complete_array = complete_array.astype(float)

    # for i in range(0, complete_array.shape[1]):
    #     column = complete_array[:, i]
    #     maxC = np.amax(column)
    #     minC = np.amin(column)
    #     if maxC > minC:
    #         complete_array[:, i] = ((column - minC) / (maxC - minC)) * (1 - (0)) + (0)

    # shuffler = np.random.permutation(complete_array.shape[0])
    # complete_array = complete_array[shuffler]
    # input_times = np.array(input_times)[shuffler]

    tensor_games = torch.Tensor(complete_array)
    tensor_times = torch.Tensor(input_times)
    # for i in range(134336):
    #     if tensor_times[i]>30:
    #         tensor_times[i]=30
    my_dataset = data.TensorDataset(tensor_games, tensor_times)

    model = NeuralNetwork()
    training_data, validation_data = data.random_split(my_dataset, [307200, 80960])
    train_loader = data.DataLoader(training_data, batch_size=64, shuffle=True)
    val_loader = data.DataLoader(validation_data, batch_size=64, shuffle=False)
    learning_rate = 0.001
    epochs = 10
    optimizer = optim.Adam(model.parameters(), learning_rate)
    criterion = nn.MSELoss()

    sum = 0
    for i in training_data:
        sum += int(i[1])
    avg = sum / len(training_data)

    def train_model(model, optimizer, criterion, epochs, train_loader, val_loader):
        train_losses, val_losses, avg_losses = [], [], []
        for e in range(epochs):
            running_loss = 0
            running_val_loss = 0
            running_avg_loss = 0
            for images, labels in train_loader:
                # Training pass
                model.train()
                optimizer.zero_grad()
                output = model.forward(images)
                new_labels = np.reshape(labels, (64, 1))
                loss = criterion(output, new_labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
            else:
                val_loss = 0
                # 6.2 Evalaute model on validation at the end of each epoch.
                with torch.no_grad():
                    for images, labels in val_loader:
                        output = model.forward(images)
                        new_labels = np.reshape(labels, (64, 1))
                        # if e == 2:
                        #     for image,i,j in zip (images,output,new_labels):
                        #         print(str(image)+" "+str(i[0])+" "+str(j[0]))
                        avg_tensor = torch.tensor(np.full((64, 1), avg))
                        avg_loss = criterion(avg_tensor, new_labels)
                        val_loss = criterion(output, new_labels)
                        running_val_loss += val_loss.item()
                        running_avg_loss += avg_loss.item()

                # 7. track train loss and validation loss
            train_losses.append(running_loss / len(train_loader))
            val_losses.append(running_val_loss / len(val_loader))
            avg_losses.append(running_avg_loss / len(val_loader))

            print("Epoch: {}/{}.. ".format(e + 1, epochs),
                  "Training Loss: {:.3f}.. ".format(running_loss / len(train_loader)),
                  "Validation Loss: {:.3f}.. ".format(running_val_loss / len(val_loader)),
                  "Average Loss: {:.3f}.. ".format(running_avg_loss / len(val_loader)))

        return train_losses, val_losses

    train_losses, val_losses = train_model(model, optimizer, criterion, epochs, train_loader, val_loader)
    list = []
    for i in range(10):
        list.append(i)
    plt.plot(list, train_losses)
    plt.plot(list, val_losses)
    plt.show()


nnTrain('threats.txt', 'available_moves.txt', 'clock.txt', 'taken.txt', 'count_moves.txt', 'materials.txt', 'times.txt')
