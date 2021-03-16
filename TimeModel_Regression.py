import numpy as np
from torch import nn, optim
from torch.utils import data
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


class NeuralNetwork(nn.Module):
    def __init__(self):
        """
        Declare layers for the model
        """
        super().__init__()
        self.fc1 = nn.Linear(6, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 32)
        self.fc6 = nn.Linear(32, 1)
        self.norm1 = nn.BatchNorm1d(512)
        self.norm2 = nn.BatchNorm1d(256)
        self.norm3 = nn.BatchNorm1d(128)
        self.norm4 = nn.BatchNorm1d(64)
        self.norm5 = nn.BatchNorm1d(32)
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
        x = self.norm3(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc4(x)
        x = self.norm4(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc5(x)
        x = self.norm5(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc6(x)
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

    threat_array = np.array(input_threats)
    moves_available_array = np.array(input_moves_available)
    clock_array = np.array(input_clock)
    move_num_array = np.array(input_move_num)
    taken_array = np.array(input_taken)
    input_materials_array = np.array(input_materials)
    complete_array = np.vstack(
        (threat_array, moves_available_array, clock_array, taken_array, move_num_array,
         input_materials_array)).transpose()
    complete_array = complete_array.astype(float)

    for i in range(0, complete_array.shape[1]):
        column = complete_array[:, i]
        maxC = np.amax(column)
        minC = np.amin(column)
        if maxC > minC:
            complete_array[:, i] = ((column - minC) / (maxC - minC)) * (1 - (-1)) + (-1)

    shuffler = np.random.permutation(complete_array.shape[0])
    complete_array = complete_array[shuffler]
    input_times = np.array(input_times)[shuffler]

    tensor_games = torch.Tensor(complete_array)
    tensor_times = torch.Tensor(input_times)

    my_dataset = data.TensorDataset(tensor_games, tensor_times)
    train_size = int(0.85 * len(my_dataset))
    validate_size = len(my_dataset) - int(0.85 * len(my_dataset))
    model = NeuralNetwork()
    training_data, validation_data = data.random_split(my_dataset, [train_size, validate_size])
    train_loader = data.DataLoader(training_data, batch_size=64, shuffle=True)
    val_loader = data.DataLoader(validation_data, batch_size=64, shuffle=False)
    learning_rate = 0.0002
    epochs = 30
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
                labels = torch.unsqueeze(labels, 1)
                labels = labels.type(torch.FloatTensor)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
            else:
                # 6.2 Evalaute model on validation at the end of each epoch.
                with torch.no_grad():
                    for images, labels in val_loader:
                        output = model.forward(images)
                        labels = torch.unsqueeze(labels, 1)
                        labels = labels.type(torch.FloatTensor)
                        if e == epochs - 1:
                            for image, i, j in zip(images, output, labels):
                                print(str(int(i)) + " " + str(int(j)))
                        avg_tensor = torch.full_like(labels, avg)
                        avg_loss = criterion(avg_tensor, labels)
                        val_loss = criterion(output, labels)
                        running_val_loss += val_loss.item()
                        running_avg_loss += avg_loss.item()

                # 7. track train loss and validation loss
            train_losses.append(running_loss / len(train_loader))
            val_losses.append(running_val_loss / len(val_loader))
            avg_losses.append(running_avg_loss / len(val_loader))

            print("Epoch: {}/{}.. ".format(e + 1, epochs),
                  "Training Loss: {:.3f}.. ".format(running_loss / len(train_loader)),
                  "Validation Loss: {:.3f}.. ".format(running_val_loss / len(val_loader)),
                  "Average Loss: {:.3f}.. ".format(running_avg_loss / len(val_loader)
                                                   ))

        return train_losses, val_losses, avg_losses

    train_losses, val_losses, avg_losses = train_model(model, optimizer, criterion, epochs, train_loader, val_loader)
    list = []
    for i in range(epochs):
        list.append(i)
    plt.plot(list, train_losses)
    plt.plot(list, val_losses)
    plt.plot(list, avg_losses)
    plt.show()


nnTrain('atleast2400s_400diff_above2000elo/masters_threats.txt',
        'atleast2400s_400diff_above2000elo/masters_available_moves.txt',
        'atleast2400s_400diff_above2000elo/masters_clock.txt', 'atleast2400s_400diff_above2000elo/masters_taken.txt',
        'atleast2400s_400diff_above2000elo/masters_moves.txt',
        'atleast2400s_400diff_above2000elo/masters_materials.txt',
        'atleast2400s_400diff_above2000elo/masters_times.txt')
