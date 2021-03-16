from chess.pgn import read_game


def number_of_moves_in_game(game):
    moves = game.mainline_moves()
    counter = 0
    for move in moves:
        counter += 1
    return counter


full_pgn = open("atleast2400s_400diff_above2000elo/atleast2400sec_400diff_above2000.pgn")
new_pgn = open("atleast2400s_400diff_above2000elo/masters3.pgn", "w")

game = read_game(full_pgn)
while game is not None:
    if number_of_moves_in_game(game) > 22:
        new_pgn.write(game.__str__())
        new_pgn.write("\n\n")
    game = read_game(full_pgn)

full_pgn.close()
new_pgn.close()
