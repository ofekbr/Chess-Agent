import chess.pgn

# engine = chess.engine.SimpleEngine.popen_uci("stockfish")
first_game_cut = 0
last_game_cut = 20
min_move_in_game = 20
gamefile = "masters3.pgn"
bool_all_but_2 = True

from enum import Enum


class Color(Enum):
    White = True
    Black = False
    empty = None


def number_of_moves_in_game(game):
    moves = game.mainline_moves()
    counter = 0
    for move in moves:
        counter += 1
    if (counter < 20):
        print(game)
    return counter


def read_relevant_game(game, pgn):
    global last_game_cut
    if game is None:
        return None
    while number_of_moves_in_game(game) < min_move_in_game:
        game = chess.pgn.read_game(pgn)
    # ONY IF WE RUN ONE AT A TIME !!!
    if bool_all_but_2:
        last_game_cut = number_of_moves_in_game(game) - 2
    return game


def print_to_file(array,file_name ):
    print(f"\n --------------------- done parsing ----------------------\n")
    print(f"starting to write the file :")
    f = open(file_name, "w")
    for i, data in enumerate(array):
        f.write(str(data) + '\n')
    f.close()


def parse_moves_available():
    input_moves_available = []
    game_number = 0
    with open(gamefile) as pgn:
        game = read_relevant_game(chess.pgn.read_game(pgn), pgn)

        while game:
            try:
                board = game.board()
                temp_moves_available = []

                for i, move in enumerate(game.mainline_moves()):
                    if i < first_game_cut:
                        board.push(move)
                        continue

                    temp_moves_available.append(board.legal_moves.count())
                    board.push(move)

                    if i >= last_game_cut - 1:
                        input_moves_available += temp_moves_available
                        break
            except:
                pass

            print(f" game number {game_number}")
            game_number += 1

            game = read_relevant_game(chess.pgn.read_game(pgn), pgn)

    print_to_file(input_moves_available,"masters_available_moves.txt")


def parse_threats():
    input_threats = []
    game_number = 0
    with open(gamefile) as pgn:
        game = read_relevant_game(chess.pgn.read_game(pgn),pgn)

        while game:
            try:
                board = game.board()
                temp_threats = []
                # for every move count the num of threats
                for k, move in enumerate(game.mainline_moves()):
                    if k < first_game_cut:
                        board.push(move)
                        continue
                    sum = 0
                    turn = Color(board.turn)
                    for i in range(8):
                        for j in range(8):
                            # on white's turn, if the pawn in the square is white check the black attackers
                            color = Color(board.color_at(chess.square(j, i)))
                            if ((color == Color.White) and (turn == Color.White)):
                                # get the number of black attackers
                                sum += len(board.attackers(0, chess.square(j, i)))
                            elif ((color == Color.Black) and (turn == Color.Black)):
                                sum += len(board.attackers(1, chess.square(j, i)))

                    temp_threats.append(sum)
                    board.push(move)

                    if k >= last_game_cut - 1:
                        input_threats += temp_threats
                        break
            except:
                pass
            print(f" game number {game_number}")
            game_number += 1
            game = read_relevant_game(chess.pgn.read_game(pgn),pgn)

    print_to_file(input_threats, "masters_threats.txt")


def parse_clock():
    input_clock = []
    game_number = 0
    with open(gamefile) as pgn:
        game = read_relevant_game(chess.pgn.read_game(pgn),pgn)
        while game:
            try:
                temp_clock = []
                for i,node in enumerate(game.mainline()):
                    if i < first_game_cut:
                        continue

                    temp_clock.append(node.clock())

                    if i >= last_game_cut - 1:
                        input_clock += temp_clock
                        break

            except:
                pass
            print(f" game number {game_number}")
            game_number += 1
            game = read_relevant_game(chess.pgn.read_game(pgn), pgn)

    print_to_file(input_clock, "masters_clock.txt")


def parse_times():
    input_times = []
    game_number=0
    with open(gamefile) as pgn:
        game = read_relevant_game(chess.pgn.read_game(pgn),pgn)
        while game:
            try:
                temp_times = []
                for i, node in enumerate(game.mainline()):
                    if i < first_game_cut - 1:
                        continue
                    if node.next() and node.next().next():
                        temp_times.append(node.clock() - node.next().next().clock())

                    if (i == last_game_cut - 1):
                        input_times += temp_times
                        break
            except:
                pass
            print(f" game number {game_number}")
            game_number += 1
            game = read_relevant_game(chess.pgn.read_game(pgn), pgn)

    print_to_file(input_times,"masters_times.txt")


def parse_taken():
    input_taken = []
    game_number = 0
    with open(gamefile) as pgn:
        game = read_relevant_game(chess.pgn.read_game(pgn),pgn)
        while game:
            try:
                temp_taken = []
                move_list = game.mainline().__str__().split('}')
                for i in range(first_game_cut, last_game_cut):
                    if len(move_list) > i and 'x' in move_list[i]:
                        temp_taken.append(1)
                    elif len(move_list) > i:
                        temp_taken.append(0)
                    else:
                        break

                input_taken += temp_taken
            except:
                pass
            print(f" game number {game_number}")
            game_number += 1
            game = read_relevant_game(chess.pgn.read_game(pgn),pgn)

    print_to_file(input_taken, "masters_taken.txt")


def parse_moves_num():
    moves_num = []
    game_num = 0

    with open(gamefile) as pgn:
        game = read_relevant_game(chess.pgn.read_game(pgn),pgn)
        while game:
            try:
                temp_count = []
                for i,_ in enumerate(game.mainline()):
                    if i < first_game_cut:
                        continue
                    temp_count.append(i)
                    if i == last_game_cut - 1:
                        break

                moves_num += temp_count
            except:
                pass
            print("game: ", game_num)
            game_num += 1
            game = read_relevant_game(chess.pgn.read_game(pgn), pgn)

    print_to_file(moves_num,"masters_moves.txt")


def parse_material_diff():
    pieces_dict = {
        'r': 1276, 'R': 1276,
        'n': 781, 'N': 781,
        'b': 825, 'B': 825,
        'q': 2538, 'Q': 2538,
        'p': 124, 'P': 124,
    }
    input_values = []
    game_num = 0
    with open(gamefile) as pgn:
        game = read_relevant_game(chess.pgn.read_game(pgn),pgn)
        while game:
            temp_count = []
            white_turn = True
            board = game.board()
            for i, move in enumerate(game.mainline_moves()):
                if i < first_game_cut:
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
                if i == last_game_cut - 1:
                    input_values += temp_count
                    break

            print(f"game number {game_num}")
            game_num += 1

            game = read_relevant_game(chess.pgn.read_game(pgn),pgn)


    print_to_file(input_values, "masters_materials.txt")


def parse_times_of_enemy():
    input_times = []
    game_number = 0
    with open(gamefile) as pgn:
        game = read_relevant_game(chess.pgn.read_game(pgn),pgn)
        while game:
            try:
                temp_times = []
                count = 10
                for i, node in enumerate(game.mainline()):
                    if i < first_game_cut:
                        continue
                    if node.next() and node.next().next():
                        temp_times.append(node.clock() - node.next().next().clock())
                        count += 1
                    if count == last_game_cut:
                        break
                temp_times.pop()
                input_times += temp_times
            except:
                pass
            print(f" game number {game_number}")
            game_number += 1
            game = read_relevant_game(chess.pgn.read_game(pgn), pgn)

    print_to_file(input_times, "masters_times_of_enemy.txt")


# parse_times_of_enemy() # 26339 ---
# parse_times() #28780 done 214246
# parse_clock() #28780 done 214246
# parse_taken() #28780 done 214246
# parse_threats()  #28780 done 214246
# parse_material_diff() #28780 done 214246
# parse_moves_num() #28780 done 214246
# parse_moves_available() #28780 done 214246



def parse_position():
    pass
    # input_mean = []
    # input_std = []
    #
    # with open(gamefile) as pgn:
    #     game = chess.pgn.read_game(pgn)
    #     start = timer()
    #     i = 0
    #     skipper = 0
    #     while game:
    #         try:
    #             board = game.board()
    #             temp_mean = []
    #             temp_std = []
    #
    #             white_turn = True
    #             for move in game.mainline_moves():
    #                 if skipper < early_game_cut:
    #                     board.push(move)
    #                     skipper += 1
    #                     continue
    #                 move_scores = []
    #                 for el in list(board.legal_moves):
    #                     info = engine.analyse(board, chess.engine.Limit(time=0.000001), root_moves=[el])
    #                     t = info["score"].relative.cp
    #                     # if t.startswith('#'):
    #                     #     print(" eval = mate in ", t)
    #                     # else:
    #                     if not white_turn:
    #                         t = -t
    #                     move_scores.append(round(t / 100., 2))
    #                 move_scores = np.array(move_scores)
    #                 moves_std = np.std(move_scores)
    #                 moves_mean = np.mean(move_scores)
    #
    #                 temp_mean += moves_mean
    #                 temp_std += moves_std
    #                 white_turn = not white_turn
    #                 board.push(move)
    #
    #             temp_mean = temp_mean[:-2]
    #             temp_std = temp_std[:-2]
    #             input_mean += temp_mean
    #             input_std += temp_std
    #         except:
    #             pass
    #         end = timer()
    #         print(f" game number {i} out of 2401, it took {timedelta(seconds=end - start)}")
    #         start = timer()
    #         i += 1
    #         game = chess.pgn.read_game(pgn)
    #         skipper = 0
    #
    # print(f"\n --------------------- done parsing ----------------------\n")
    # print(f"starting to write the file :")
    # f = open("std.txt", "w")
    # for i, data in enumerate(input_std):
    #     f.write(str(data) + '\n')
    #     print(f"printing data number {i}")
    # f.close()
    #
    # f = open("mean.txt", "w")
    # for i, data in enumerate(input_mean):
    #     f.write(str(data) + '\n')
    #     print(f"printing data number {i}")
    # f.close()
