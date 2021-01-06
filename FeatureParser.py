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

# engine = chess.engine.SimpleEngine.popen_uci("stockfish")
early_game_cut = 12
gamefile = "1800sec_400_2000.pgn"


def parse_moves_available():
    input_moves_available = []
    with open(gamefile) as pgn:
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
    with open(gamefile) as pgn:
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
    with open(gamefile) as pgn:
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
    with open(gamefile) as pgn:
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
    with open(gamefile) as pgn:
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

    with open(gamefile) as pgn:
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
    with open(gamefile) as pgn:
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


def parse_material_diff():
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
    with open(gamefile) as pgn:
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

parse_times()
parse_clock()
parse_taken()
parse_threats()
parse_material_diff()
parse_moves_num()
parse_moves_available()
