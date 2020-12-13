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
        self.fc1 = nn.Linear(7, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 1)
        self.norm1 = nn.BatchNorm1d(128)
        self.norm2 = nn.BatchNorm1d(32)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.norm1(x)
        x = F.relu(self.fc2(x))
        x = self.norm2(x)
        x = self.fc3(x)
        return x


engine = chess.engine.SimpleEngine.popen_uci("stockfish")
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


def parse_taken():
    input_taken = []
    with open("classic_600+0_200_1800.pgn") as pgn:
        game = chess.pgn.read_game(pgn)
        skipper = 0
        while game:
            try:
                board = game.board()
                temp_taken = []
                move_list=game.mainline().__str__().split('}')
                for i in range(12,len(move_list)):
                    if 'x' in move_list[i]:
                        temp_taken.append(1)
                    else:
                        temp_taken.append(0)


                temp_taken = temp_taken[:-3]
                input_taken += temp_taken
            except Exception as inst:
                print(inst.__class__)
                pass
            game = chess.pgn.read_game(pgn)
            skipper = 0

    print(f"\n --------------------- done parsing ----------------------\n")
    print(f"starting to write the file :")
    f = open("taken.txt", "w")
    for i, data in enumerate(input_taken):
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


def parse_moves_num():
    moves_num = []
    with open("classic_600+0_200_1800.pgn") as pgn:
        game = chess.pgn.read_game(pgn)
        skipper = 0
        i = 1
        while game:
            print("game: ", i)
            i=i+1
            try:
                board = game.board()
                temp_count = []
                count = early_game_cut
                for node in game.mainline():
                    if skipper < early_game_cut:
                        skipper += 1
                        continue
                    if node.next() and node.next().next():
                        count = count + 1
                        temp_count.append(count)

                moves_num += temp_count
            except:
                pass
            game = chess.pgn.read_game(pgn)
            skipper = 0

    print(f"\n --------------------- done parsing ----------------------\n")
    print(f"starting to write the file :")
    f = open("count_moves.txt", "w")
    for i, data in enumerate(moves_num):
        f.write(str(data) + '\n')
        print(f"printing data number {i}")
    f.close()


def material_diff():
    skipper = 0
    pieces_dict = {
        'r': 1276, 'R': 1276,
        'n': 781, 'N': 781,
        'b': 825, 'B': 825,
        'q': 2538, 'Q': 2538,
        'p': 124, 'P': 124,
    }
    input_values = []
    i = 0
    with open("classic_600+0_200_1800.pgn") as pgn:
        game = chess.pgn.read_game(pgn)
        while game:
            i+=1
            temp_count = []
            print(f"game number {i} out of 2401")
            white_turn = True
            board = game.board()
            for move in game.mainline_moves():
                if skipper < early_game_cut:
                    board.push(move)
                    skipper += 1
                    continue
                white_sum = 0
                black_sum = 0
                for char in str(board):
                    if char in ['\n', '.', ' ', '1', '2', '3', '4', '5', '6', '7', '8']:
                        continue
                    if char in ['r', 'n', 'b', 'q', 'p']:
                        black_sum += pieces_dict[char]
                    if char in ['R', 'N', 'B', 'Q', 'P']:
                        white_sum += pieces_dict[char]
                if white_turn:
                    temp_count.append(white_sum - black_sum)
                else:
                    temp_count.append(black_sum - white_sum)
                white_turn = not white_turn
                board.push(move)
            temp_count = temp_count[:-2]
            input_values += temp_count
            skipper = 0
            game = chess.pgn.read_game(pgn)

        print(len(input_values))
        print(f"\n --------------------- done parsing ----------------------\n")
        print(f"starting to write the file :")
        f = open("materials.txt", "w")
        for i, data in enumerate(input_values):
            f.write(str(data) + '\n')
            print(f"printing data number {i}")
        f.close()


def nnTrain(threats_path,moves_available_path,clock_path,taken_path,move_num_path,materials_path,times_path):
    torch.manual_seed(0)

    input_threats=np.loadtxt(threats_path,dtype=int).tolist()
    input_moves_available=np.loadtxt(moves_available_path,dtype=int).tolist()
    input_clock=np.loadtxt(clock_path,dtype=int).tolist()
    input_taken=np.loadtxt(taken_path).tolist()
    input_move_num=np.loadtxt(move_num_path,dtype=int).tolist()
    input_materials = np.loadtxt(materials_path, dtype=int).tolist()
    input_times=np.loadtxt(times_path,dtype=int)

    print(len(input_threats))
    print(len(input_moves_available))
    print(len(input_clock))
    print(len(input_taken))
    print(len(input_move_num))
    print(len(input_materials))
    print(len(input_times))


    input_threats = input_threats[:-29]
    input_moves_available = input_moves_available[:-29]
    input_clock = input_clock[:-29]
    input_times = input_times[:-29]
    input_move_num = input_move_num[:-29]
    input_materials = input_materials[:-29]
    input_taken = input_taken[:-29]

    threat_array = np.array(input_threats)
    moves_available_array = np.array(input_moves_available)
    clock_array = np.array(input_clock)
    move_num_array = np.array(input_move_num)
    taken_array = np.array(input_taken)
    input_materials = np.array(input_materials)
    complete_array = np.vstack((threat_array, moves_available_array, clock_array,taken_array,move_num_array,input_materials)).transpose()
    complete_array = np.insert(complete_array, 0, values=1, axis=1) # Insert values before column 1
    complete_array = complete_array.astype(float)

    for i in range(1, complete_array.shape[1]):
        column = complete_array[:, i]
        maxC = np.amax(column)
        minC = np.amin(column)
        if maxC > minC:
            complete_array[:, i] =((column - minC) / (maxC - minC)) * (1 - (0)) + (0)

    #shuffler = np.random.permutation(complete_array.shape[0])
    #complete_array = complete_array[shuffler]
    #input_times = np.array(input_times)[shuffler]

    tensor_games = torch.Tensor(complete_array)
    tensor_times = torch.Tensor(input_times)

    my_dataset = data.TensorDataset(tensor_games, tensor_times)

    model = NeuralNetwork()
    training_data, validation_data = data.random_split(my_dataset, [108800, 25536])
    train_loader = data.DataLoader(training_data, batch_size=64, shuffle=True)
    val_loader = data.DataLoader(validation_data, batch_size=64, shuffle=False)
    learning_rate = 0.0001
    epochs = 30
    optimizer = optim.Adam(model.parameters(), learning_rate)
    criterion = nn.MSELoss()

    sum=0
    for i in validation_data:
        sum+=int(i[1])
    avg=sum/len(validation_data)

    def train_model(model, optimizer, criterion, epochs, train_loader, val_loader):
        train_losses, val_losses, avg_losses = [], [], []
        for e in range(epochs):
            running_loss = 0
            running_val_loss = 0
            running_avg_loss=0
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
                        # if e == 14:
                        #     for i,j in zip (output,new_labels):
                        #         print(str(i[0])+" "+str(j[0]))
                        avg_tensor=torch.tensor(np.full((64,1),avg))
                        # sum=0
                        # for i in new_labels:
                        #     sum+=(i-avg)**2
                        # sum/=64
                        avg_loss = criterion(avg_tensor, new_labels)
                        val_loss = criterion(output, new_labels)
                        running_val_loss += val_loss.item()
                        running_avg_loss += avg_loss.item()

                # 7. track train loss and validation loss
            train_losses.append(running_loss / len(train_loader))
            val_losses.append(running_val_loss / len(val_loader))
            avg_losses.append(running_avg_loss/len(val_loader))


            print("Epoch: {}/{}.. ".format(e + 1, epochs),
                   "Training Loss: {:.3f}.. ".format(running_loss / len(train_loader)),
                   "Validation Loss: {:.3f}.. ".format(running_val_loss / len(val_loader)),
                  "Average Loss: {:.3f}.. ".format(running_avg_loss / len(val_loader)))

        return train_losses, val_losses


    train_losses, val_losses = train_model(model, optimizer, criterion, epochs, train_loader, val_loader)
    list = []
    for i in range(30):
        list.append(i)
    plt.plot(list, train_losses)
    plt.plot(list, val_losses)
    plt.show()

nnTrain('threats.txt','available_moves.txt','clock.txt','taken.txt','count_moves.txt','materials.txt','times.txt')

