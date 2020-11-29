import chess.pgn
import numpy as np
from torch import nn, optim
from torch.utils import data
import torch
import matplotlib.pyplot as plt

engine = chess.engine.SimpleEngine.popen_uci("stockfish")

class NeuralNetwork(nn.Module):
    def __init__(self):
        """
        Declare layers for the model
        """
        super().__init__()
        self.fc1 = nn.Linear(1536, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 8)
        self.fc4 = nn.Linear(8, 1)

    def forward(self, x):
        """
        Forward pass through the network, returns log_softmax values
        """
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        return x


input_games = []
input_times = []
input_moves = []
#input_scores = []


def split(word):
    return [char for char in word]


with open("test_games.pgn") as pgn:
    game = chess.pgn.read_game(pgn)
    pieces_dict = {'P': 0, 'R': 1, 'Q': 2, 'B': 3, 'N': 4, 'K': 5, '.': -1,
                   'p': 6, 'r': 7, 'q': 8, 'b': 9, 'n': 10, 'k': 11}
    while game:
        try:
            board = game.board()
            temp_games = []
            temp_moves = []
            temp_times = []
            #temp_scores= []

            for move in game.mainline_moves():
                #try:
                #    info = engine.analyse(board, chess.engine.Limit(time=0.1))
                #    temp_scores.append(info['score'].relative.cp)
                #except:
                #    pass
                temp_char_list = split(board.__str__())
                temp_list = []
                init_board = np.zeros((24, 8, 8))
                row, col = 0, 0
                for char in temp_char_list:
                    if char == ' ' or char == '.':
                        if(char == '.'):
                            col+=1
                    elif char == '\n':
                        row += 1
                        col = 0
                        continue
                    else:
                        init_board[pieces_dict[char]][row][col] = 1
                        col += 1
                        # temp_list.append(pieces_dict[char])
                temp_games.append(init_board)
                temp_moves.append(move.__str__())
                board.push(move)

            for node in game.mainline():
                if node.next() and node.next().next():
                    temp_times.append(node.clock() - node.next().next().clock())

            for i in range(1, len(temp_games)):
                for j in range(12):
                    temp_games[i-1][j+12] = temp_games[i][j]
            temp_games = temp_games[:-2]
            temp_moves = temp_moves[:-2]
            input_moves += temp_moves
            input_games += temp_games
            input_times += temp_times
            #input_scores += temp_scores
        except:
            pass
        game = chess.pgn.read_game(pgn)
print(len(input_moves))
print(len(input_games))
print(len(input_times))
#print(len(input_scores))

input_moves = input_moves[:-48]
input_games = input_games[:-48]
input_times = input_times[:-48]
"""
min_game = min(input_games)
max_game = max(input_games)
min_time = min(input_times)
max_time = max(input_times)

global_min=999999
global_max=0
for i in range(len(input_games)):
    min_game = min(min(input_games[i]))
    max_game = max(max(input_games[i]))
    if min_game<global_min:
        global_min=min_game
    if max_game>global_max:
        global_max=max_game

for i in range(len(input_games)):
    for j in range(i):

for i in range(len(input_times)):
    v = input_times[i]   # foo[:, -1] for the last column
    input_times[i] = (v - min_time) / (max_time - min_time)
"""
npArray = np.array(input_games)
tensor_games= np.reshape(npArray, (11200,1536))
tensor_games = torch.Tensor(tensor_games)
# tensor_moves = torch.Tensor(input_moves)
tensor_times = torch.Tensor(input_times)

# transforms.Normalize(0, 1)(tensor_times)
# transforms.Normalize(0, 1)(tensor_games)

my_dataset = data.TensorDataset(tensor_games, tensor_times)
my_dataloader = data.DataLoader(my_dataset)

model = NeuralNetwork()
training_data, validation_data = data.random_split(my_dataset, [9024, 2176])
train_loader = data.DataLoader(training_data, batch_size=64, shuffle=True)
val_loader = data.DataLoader(validation_data, batch_size=64, shuffle=False)
learning_rate = 0.00001
epochs = 50
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
for i in range(50):
    list.append(i)
plt.plot(list, train_losses)
plt.plot(list, val_losses)
plt.show()
