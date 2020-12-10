import chess.pgn
import numpy as np
from matplotlib import transforms
from torch import nn, optim
from torch.utils import data
import torch
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
        self.fc1 = nn.Linear(4, 1)

    def forward(self, x):
        """
        Forward pass through the network, returns log_softmax values
        """
        x = (self.fc1(x))
        return x


engine = chess.engine.SimpleEngine.popen_uci("stockfish_20090216_x64")
early_game_cut = 12


def parse_moves_available():
    input_moves_available = []
    with open("classic_600+0_200_1800.pgn") as pgn:
        game = chess.pgn.read_game(pgn)
        i = 0
        skipper = 0
        while game:
            try:
                board = game.board()
                temp_moves_available = []
                for move in game.mainline_moves():
                    if skipper < early_game_cut:
                        board.push(move)
                        skipper += 1
                        continue

                    temp_moves_available.append(board.legal_moves.count())
                    board.push(move)

                temp_moves_available = temp_moves_available[:-2]
                input_moves_available += temp_moves_available

            except:
                pass
            print(f" game number {i} out of 2401")
            i += 1
            game = chess.pgn.read_game(pgn)
            skipper = 0

    print(f"\n --------------------- done parsing ----------------------\n")
    print(f"starting to write the file :")
    f = open("available_moves.txt", "w")
    for i, data in enumerate(input_moves_available):
        f.write(str(data) + '\n')
        print(f"printing data number {i}")
    f.close()


def parse_threats():
    input_threats = []
    with open("classic_600+0_200_1800.pgn") as pgn:
        game = chess.pgn.read_game(pgn)
        skipper = 0
        while game:
            try:
                board = game.board()
                temp_threats = []
                for move in game.mainline_moves():
                    if skipper < early_game_cut:
                        board.push(move)
                        skipper += 1
                        continue
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

                temp_threats = temp_threats[:-2]
                input_threats += temp_threats
            except:
                pass
            game = chess.pgn.read_game(pgn)
            skipper = 0

    print(f"\n --------------------- done parsing ----------------------\n")
    print(f"starting to write the file :")
    f = open("threats.txt", "w")
    for i, data in enumerate(input_threats):
        f.write(str(data) + '\n')
        print(f"printing data number {i}")
    f.close()


def parse_clock():
    input_clock = []
    with open("classic_600+0_200_1800.pgn") as pgn:
        game = chess.pgn.read_game(pgn)
        skipper = 0
        while game:
            try:
                board = game.board()

                temp_clock = []
                for node in game.mainline():
                    if skipper < early_game_cut:
                        skipper += 1
                        continue
                    if node.next() and node.next().next():
                        temp_clock.append(node.clock())

                input_clock += temp_clock
            except:
                pass
            game = chess.pgn.read_game(pgn)
            skipper = 0

    print(f"\n --------------------- done parsing ----------------------\n")
    print(f"starting to write the file :")
    f = open("clock.txt", "w")
    for i, data in enumerate(input_clock):
        f.write(str(data) + '\n')
        print(f"printing data number {i}")
    f.close()


def parse_times():
    input_times = []
    with open("classic_600+0_200_1800.pgn") as pgn:
        game = chess.pgn.read_game(pgn)
        skipper = 0
        while game:
            try:
                temp_times = []
                for node in game.mainline():
                    if skipper < early_game_cut:
                        skipper += 1
                        continue
                    if node.next() and node.next().next():
                        temp_times.append(node.clock() - node.next().next().clock())

                input_times += temp_times
            except:
                pass
            game = chess.pgn.read_game(pgn)
            skipper = 0
    print(f"\n --------------------- done parsing ----------------------\n")
    print(f"starting to write the file :")
    f = open("times.txt", "w")
    for i, data in enumerate(input_times):
        f.write(str(data) + '\n')
        print(f"printing data number {i}")
    f.close()


def parse_position():
    input_mean = []
    input_std = []

    with open("classic_600+0_200_1800.pgn") as pgn:
        game = chess.pgn.read_game(pgn)
        start = timer()
        i = 0
        skipper = 0
        while game:
            try:
                board = game.board()
                temp_mean = []
                temp_std = []

                white_turn = True
                for move in game.mainline_moves():
                    if skipper < early_game_cut:
                        board.push(move)
                        skipper += 1
                        continue
                    move_scores = []
                    for el in list(board.legal_moves):
                        info = engine.analyse(board, chess.engine.Limit(time=0.000001), root_moves=[el])
                        t = info["score"].relative.cp
                        # if t.startswith('#'):
                        #     print(" eval = mate in ", t)
                        # else:
                        if not white_turn:
                            t = -t
                        move_scores.append(round(t / 100., 2))
                    move_scores = np.array(move_scores)
                    moves_std = np.std(move_scores)
                    moves_mean = np.mean(move_scores)

                    temp_mean += moves_mean
                    temp_std += moves_std
                    white_turn = not white_turn
                    board.push(move)

                temp_mean = temp_mean[:-2]
                temp_std = temp_std[:-2]
                input_mean += temp_mean
                input_std += temp_std
            except:
                pass
            end = timer()
            print(f" game number {i} out of 2401, it took {timedelta(seconds=end - start)}")
            start = timer()
            i += 1
            game = chess.pgn.read_game(pgn)
            skipper = 0

    print(f"\n --------------------- done parsing ----------------------\n")
    print(f"starting to write the file :")
    f = open("std.txt", "w")
    for i, data in enumerate(input_std):
        f.write(str(data) + '\n')
        print(f"printing data number {i}")
    f.close()

    f = open("mean.txt", "w")
    for i, data in enumerate(input_mean):
        f.write(str(data) + '\n')
        print(f"printing data number {i}")
    f.close()


# parse_moves_available()
# parse_threats()
# parse_clock()
# parse_times()
parse_position()


def nnTrain(threats_path,moves_available_path,clock_path,std_path,mean_path,times_path):
    input_threats=np.loadtxt(threats_path,dtype=int).tolist()
    input_moves_available=np.loadtxt(moves_available_path,dtype=int).tolist()
    input_clock=np.loadtxt(clock_path,dtype=int).tolist()
    input_std=np.loadtxt(std_path).tolist()
    input_mean=np.loadtxt(mean_path).tolist()
    input_times=np.loadtxt(times_path,dtype=int)

    print(len(input_threats))
    print(len(input_moves_available))
    print(len(input_clock))
    print(len(input_std))
    print(len(input_mean))
    print(len(input_times))

    input_threats = input_threats[:-13]
    input_moves_available = input_moves_available[:-13]
    input_clock = input_clock[:-13]
    input_times = input_times[:-13]

    threat_array = np.array(input_threats)
    moves_available_array = np.array(input_moves_available)
    clock_array = np.array(input_clock)
    complete_array = np.vstack((threat_array, moves_available_array, clock_array)).transpose()
    complete_array = np.insert(complete_array, 0, values=1, axis=1) # Insert values before column 3
    complete_array = complete_array.astype(float)

    for i in range(1, complete_array.shape[1]):
        column = complete_array[:, i]
        maxC = np.amax(column)
        minC = np.amin(column)
        if maxC > minC:
            complete_array[:, i] =((column - minC) / (maxC - minC)) * (1 - (0)) + (0)

    #shuffler = np.random.permutation(npArray3.shape[0])
    #npArray3 = npArray3[shuffler]
    #input_times = np.array(input_times)[shuffler]

    tensor_games = torch.Tensor(complete_array)
    tensor_times = torch.Tensor(input_times)

    my_dataset = data.TensorDataset(tensor_games, tensor_times)

    model = NeuralNetwork()
    training_data, validation_data = data.random_split(my_dataset, [128000, 35008])
    train_loader = data.DataLoader(training_data, batch_size=64, shuffle=True)
    val_loader = data.DataLoader(validation_data, batch_size=64, shuffle=False)
    learning_rate = 0.0001
    epochs = 15
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
                        if e == 14:
                            for i,j in zip (output,new_labels):
                                print(str(i[0])+" "+str(j[0]))
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
    for i in range(15):
        list.append(i)
    plt.plot(list, train_losses)
    plt.plot(list, val_losses)
    plt.show()


