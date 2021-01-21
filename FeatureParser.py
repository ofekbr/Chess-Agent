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
first_game_cut = 10
last_game_cut = 20
gamefile = "masters.pgn"

def number_of_moves_in_game(game):
    moves=game.mainline_moves()
    counter=0
    for move in moves:
        counter+=1
    if (counter<20):
        print(game)
    return counter

def parse_moves_available():
    input_moves_available = []
    with open(gamefile) as pgn:
        game = chess.pgn.read_game(pgn)
        while (number_of_moves_in_game(game)<20):
            game = chess.pgn.read_game(pgn)
        i = 0
        skipper = 0
        while game:
            try:
                board = game.board()
                temp_moves_available = []
                for move in game.mainline_moves():
                    if skipper < first_game_cut:
                        board.push(move)
                        skipper += 1
                        continue

                    temp_moves_available.append(board.legal_moves.count())
                    board.push(move)

                    skipper += 1
                    if(skipper>=last_game_cut):
                        input_moves_available += temp_moves_available
                        break
            except:
                pass
            print(f" game number {i} out of 2401")
            i += 1
            game = chess.pgn.read_game(pgn)
            while (game and number_of_moves_in_game(game) < 20):
                game = chess.pgn.read_game(pgn)
            skipper = 0
    #print("moves available:{")
    print(f"\n --------------------- done parsing ----------------------\n")
    print(f"starting to write the file :")
    f = open("masters_available_moves.txt", "w")
    for i, data in enumerate(input_moves_available):
        f.write(str(data) + '\n')
        #print(f"printing data number {i}")
    f.close()


def parse_threats():
    input_threats = []
    with open(gamefile) as pgn:
        game = chess.pgn.read_game(pgn)
        skipper = 0
        k=0
        while game:
            try:
                board = game.board()
                temp_threats = []
                for move in game.mainline_moves():
                    if skipper < first_game_cut:
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
                    skipper += 1
                    if (skipper >= last_game_cut):
                        input_threats += temp_threats
                        break

            except:
                pass
            print(f" game number {k} out of 2401")
            k += 1
            game = chess.pgn.read_game(pgn)
            skipper = 0

    print(f"\n --------------------- done parsing ----------------------\n")
    print(f"starting to write the file :")
    f = open("masters_threats.txt", "w")
    for i, data in enumerate(input_threats):
        f.write(str(data) + '\n')
        print(f"printing data number {i}")
    f.close()


def parse_clock():
    input_clock = []
    with open(gamefile) as pgn:
        game = chess.pgn.read_game(pgn)
        skipper = 0
        k=0
        while game:
            try:
                temp_clock = []
                for node in game.mainline():
                    if skipper < first_game_cut-2:
                        skipper += 1
                        continue
                    if node.next() and node.next().next():
                        temp_clock.append(node.clock())

                    skipper += 1
                    if (skipper >= last_game_cut - 2):
                        input_clock += temp_clock
                        break

            except:
                pass
            print(f" game number {k} out of 2401")
            k += 1
            game = chess.pgn.read_game(pgn)
            skipper = 0


    print(f"\n --------------------- done parsing ----------------------\n")
    print(f"starting to write the file :")
    f = open("masters_clock.txt", "w")
    for i, data in enumerate(input_clock):
        f.write(str(data) + '\n')
        print(f"printing data number {i}")
    f.close()


def parse_times():
    input_times = []
    with open(gamefile) as pgn:
        game = chess.pgn.read_game(pgn)
        skipper = 0
        k = 0
        while game:
            try:
                temp_times = []
                for node in game.mainline():
                    skipper += 1
                    if skipper < first_game_cut-2:
                        #skipper += 1
                        continue
                    if node.next() and node.next().next():
                        temp_times.append(node.clock() - node.next().next().clock() + 30)

                    if (skipper == last_game_cut-2-1):
                        input_times += temp_times
                        break
            except Exception :
                print()
                pass
            print(f" game number {k}")
            k += 1
            game = chess.pgn.read_game(pgn)
            skipper = 0
    print(f"\n --------------------- done parsing ----------------------\n")
    #print(f"starting to write the file :")
    f = open("masters_times.txt", "w")
    for i, data in enumerate(input_times):
        f.write(str(data) + '\n')
        #print(f"printing data number {i}")
    f.close()


def parse_taken():
    input_taken = []
    with open(gamefile) as pgn:
        game = chess.pgn.read_game(pgn)
        while game:
            try:
                temp_taken = []
                move_list=game.mainline().__str__().split('}')
                for i in range(first_game_cut, last_game_cut):
                    if len(move_list)>i and 'x' in move_list[i]:
                        temp_taken.append(1)
                    elif len(move_list)>i:
                        temp_taken.append(0)
                    else:
                        break
                    if i == last_game_cut-1:
                        input_taken += temp_taken
            except Exception as inst:
                print(inst.__class__)
                pass
            game = chess.pgn.read_game(pgn)

    print(f"\n --------------------- done parsing ----------------------\n")
    print(f"starting to write the file :")
    f = open("masters_taken.txt", "w")
    for i, data in enumerate(input_taken):
        f.write(str(data) + '\n')
        print(f"printing data number {i}")
    f.close()


# def parse_position():
#     input_mean = []
#     input_std = []
#
#     with open(gamefile) as pgn:
#         game = chess.pgn.read_game(pgn)
#         start = timer()
#         i = 0
#         skipper = 0
#         while game:
#             try:
#                 board = game.board()
#                 temp_mean = []
#                 temp_std = []
#
#                 white_turn = True
#                 for move in game.mainline_moves():
#                     if skipper < early_game_cut:
#                         board.push(move)
#                         skipper += 1
#                         continue
#                     move_scores = []
#                     for el in list(board.legal_moves):
#                         info = engine.analyse(board, chess.engine.Limit(time=0.000001), root_moves=[el])
#                         t = info["score"].relative.cp
#                         # if t.startswith('#'):
#                         #     print(" eval = mate in ", t)
#                         # else:
#                         if not white_turn:
#                             t = -t
#                         move_scores.append(round(t / 100., 2))
#                     move_scores = np.array(move_scores)
#                     moves_std = np.std(move_scores)
#                     moves_mean = np.mean(move_scores)
#
#                     temp_mean += moves_mean
#                     temp_std += moves_std
#                     white_turn = not white_turn
#                     board.push(move)
#
#                 temp_mean = temp_mean[:-2]
#                 temp_std = temp_std[:-2]
#                 input_mean += temp_mean
#                 input_std += temp_std
#             except:
#                 pass
#             end = timer()
#             print(f" game number {i} out of 2401, it took {timedelta(seconds=end - start)}")
#             start = timer()
#             i += 1
#             game = chess.pgn.read_game(pgn)
#             skipper = 0
#
#     print(f"\n --------------------- done parsing ----------------------\n")
#     print(f"starting to write the file :")
#     f = open("std.txt", "w")
#     for i, data in enumerate(input_std):
#         f.write(str(data) + '\n')
#         print(f"printing data number {i}")
#     f.close()
#
#     f = open("mean.txt", "w")
#     for i, data in enumerate(input_mean):
#         f.write(str(data) + '\n')
#         print(f"printing data number {i}")
#     f.close()


# def parse_moves_num():
#     moves_num = []
#     with open(gamefile) as pgn:
#         game = chess.pgn.read_game(pgn)
#         skipper = 0
#         i = 1
#         while game:
#             print("game: ", i)
#             i=i+1
#             try:
#                 board = game.board()
#                 temp_count = []
#                 count = early_game_cut
#                 for node in game.mainline():
#                     if skipper < early_game_cut:
#                         skipper += 1
#                         continue
#                     if node.next() and node.next().next():
#                         count = count + 1
#                         temp_count.append(count)
#
#                 moves_num += temp_count
#             except:
#                 pass
#             game = chess.pgn.read_game(pgn)
#             skipper = 0
#
#     print(f"\n --------------------- done parsing ----------------------\n")
#     print(f"starting to write the file :")
#     f = open("count_moves.txt", "w")
#     for i, data in enumerate(moves_num):
#         f.write(str(data) + '\n')
#         print(f"printing data number {i}")
#     f.close()


def parse_material_diff():
    skipper_for_move = 0
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
            i += 1
            temp_count = []
            #print(f"game number {i} out of 2401")
            white_turn = True
            board = game.board()
            for move in game.mainline_moves():
                skipper_for_move += 1
                if skipper_for_move < first_game_cut:
                    board.push(move)
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
                if (skipper_for_move == last_game_cut-1):
                    input_values += temp_count
                    break
            skipper_for_move = 0
            game = chess.pgn.read_game(pgn)

        print(len(input_values))
        print(f"\n --------------------- done parsing ----------------------\n")
        print(f"starting to write the file :")
        f = open("masters_materials.txt", "w")
        for i, data in enumerate(input_values):
            f.write(str(data) + '\n')
            print(f"printing data number {i}")
        f.close()


# def parse_times_of_enemy():
#     input_times = []
#     with open(gamefile) as pgn:
#         game = chess.pgn.read_game(pgn)
#         skipper = 0
#         while game:
#             try:
#                 temp_times = []
#                 temp_times.append(0)
#                 for node in game.mainline():
#                     if skipper < (early_game_cut):
#                         skipper += 1
#                         continue
#                     if node.next() and node.next().next():
#                         temp_times.append(node.clock() - node.next().next().clock())
#                 temp_times.pop()
#                 input_times += temp_times
#             except:
#                 pass
#             game = chess.pgn.read_game(pgn)
#             skipper = 0
#     print(f"\n --------------------- done parsing ----------------------\n")
#     print(f"starting to write the file :")
#     f = open("times_of_enemy.txt", "w")
#     for i, data in enumerate(input_times):
#         f.write(str(data) + '\n')
#         print(f"printing data number {i}")
#     f.close()

# parse_times_of_enemy() no
parse_times() #3300 done
# parse_clock() no
# parse_taken() 3300 done
# parse_threats() 3300 done
# parse_material_diff() 3300 done
# parse_moves_num() no
# parse_moves_available() 3300 done
