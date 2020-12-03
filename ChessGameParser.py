import chess.pgn
import numpy as np
from matplotlib import transforms
from torch import nn, optim
from torch.utils import data
import torch
import matplotlib.pyplot as plt
from chess import engine

engine = chess.engine.SimpleEngine.popen_uci("stockfish")


class NeuralNetwork(nn.Module):
    def __init__(self):
        """
        Declare layers for the model
        """
        super().__init__()
        self.fc1 = nn.Linear(3, 1)

    def forward(self, x):
        """
        Forward pass through the network, returns log_softmax values
        """
        x = torch.relu(self.fc1(x))
        return x


input_moves_available = []
input_threats = []
input_times = []


def split(word):
    return [char for char in word]


with open("test_games.pgn") as pgn:
    game = chess.pgn.read_game(pgn)
    pieces_dict = {'P': 0, 'R': 1, 'Q': 2, 'B': 3, 'N': 4, 'K': 5, '.': -1,
                   'p': 6, 'r': 7, 'q': 8, 'b': 9, 'n': 10, 'k': 11}
    while game:
        try:
            board = game.board()
            temp_moves_available = []
            temp_threats = []
            temp_times = []

            for move in game.mainline_moves():
                """
                info = engine.analyse(board, chess.engine.Limit(time=1), root_moves=[move])
                t = str(info["score"])
                if t.startswith('#'):
                    print(str(move), " eval = mate in ", t)
                else:
                    print(str(move), " eval = ", round(int(t) / 100., 2))
                continue
                """
                temp_moves_available.append(board.legal_moves.count())

                sum = 0
                for i in range(8):
                    for j in range(8):
                        color = board.color_at(chess.square(j, i))
                        if color == True:
                            sum += len(board.attackers(0, chess.square(j, i)))
                        elif color == False:
                            sum += len(board.attackers(1, chess.square(j, i)))

                temp_threats.append(sum)

                board.push(move)

            for node in game.mainline():
                if node.next() and node.next().next():
                    temp_times.append(node.clock() - node.next().next().clock())

            temp_moves_available = temp_moves_available[:-2]
            temp_threats = temp_threats[:-2]
            input_threats += temp_threats
            input_moves_available += temp_moves_available
            input_times += temp_times
        except:
            pass
        game = chess.pgn.read_game(pgn)
print(len(input_threats))
print(len(input_moves_available))
print(len(input_times))

input_threats = input_threats[:-48]
input_moves_available = input_moves_available[:-48]
input_times = input_times[:-48]

npArray1 = np.array(input_threats)
npArray2 = np.array(input_moves_available)
npArray3 = np.vstack((npArray1, npArray2)).transpose()
npArray3 = np.insert(npArray3, 0, values=1, axis=1) # Insert values before column 3
npArray3 = npArray3.astype(float)

for i in range(1, npArray3.shape[1]):
    column = npArray3[:, i]
    maxC = np.amax(column)
    minC = np.amin(column)
    if maxC > minC:
        npArray3[:, i] =((column - minC) / (maxC - minC)) * (1 - (0)) + (0)

#shuffler = np.random.permutation(npArray3.shape[0])
#npArray3 = npArray3[shuffler]
#input_times = np.array(input_times)[shuffler]

tensor_games = torch.Tensor(npArray3)
# tensor_moves = torch.Tensor(input_moves)
tensor_times = torch.Tensor(input_times)


my_dataset = data.TensorDataset(tensor_games, tensor_times)
my_dataloader = data.DataLoader(my_dataset)

model = NeuralNetwork()
training_data, validation_data = data.random_split(my_dataset, [9024, 2176])
train_loader = data.DataLoader(training_data, batch_size=64, shuffle=True)
val_loader = data.DataLoader(validation_data, batch_size=64, shuffle=False)
learning_rate = 0.0001
epochs = 200
optimizer = optim.Adam(model.parameters(), learning_rate)
criterion = nn.MSELoss()


def train_model(model, optimizer, criterion, epochs, train_loader, val_loader):
    train_losses, val_losses = [], []
    for e in range(epochs):
        running_loss = 0
        running_val_loss = 0
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
                    val_loss = criterion(output, new_labels)
                running_val_loss += val_loss.item()

            # 7. track train loss and validation loss
        train_losses.append(running_loss / len(train_loader))
        val_losses.append(running_val_loss / len(val_loader))

        print("Epoch: {}/{}.. ".format(e + 1, epochs),
              "Training Loss: {:.3f}.. ".format(running_loss / len(train_loader)),
              "Validation Loss: {:.3f}.. ".format(running_val_loss / len(val_loader)))

    return train_losses, val_losses


train_losses, val_losses = train_model(model, optimizer, criterion, epochs, train_loader, val_loader)
list = []
for i in range(200):
    list.append(i)
plt.plot(list, train_losses)
plt.plot(list, val_losses)
plt.show()
