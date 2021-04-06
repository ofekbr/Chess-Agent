import chess.pgn
from enum import Enum

# engine = chess.engine.SimpleEngine.popen_uci("stockfish")

gamefile = "atleast2400s_400diff_above2000elo/atleast2400sec_400diff_above2000.pgn"


def print_to_file(array, file_name):
    print(f"\n --------------------- done parsing ----------------------\n")
    print(f"starting to write the file :")
    f = open(file_name, "w")
    for i, data in enumerate(array):
        f.write(str(data) + '\n')
    f.close()


def parse_openings():
    with open(gamefile) as pgn:
        game = chess.pgn.read_game(pgn)
        entrys = []
        while game:
            white_elo = game.headers['WhiteElo']
            opening = game.headers['Opening']
            entrys.append(white_elo+" "+opening)
            game = chess.pgn.read_game(pgn)
        print_to_file(entrys,'MastersOpenings.txt')
parse_openings()
    #print_to_file(input_moves_available, "atleast2400s_400diff_above2000elo/masters_available_moves.txt")
