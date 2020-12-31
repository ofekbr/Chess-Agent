from chess.pgn import read_game
from timeit import default_timer as timer


full_pgn = open("lichess_db_standard_rated_2017-04.pgn")
new_pgn = open("long_400_2000.pgn", "w")

game = read_game(full_pgn)
skipped, i, j = 0, 0, 0
while game is not None:
    j += 1
    headers = game.headers
    w_elo = int(headers["WhiteElo"])
    b_elo = int(headers["BlackElo"])
    time_control = headers["TimeControl"]
    if time_control == "-":
        game = read_game(full_pgn)
        continue
    seconds, addition = time_control.split("+")
    seconds = int(seconds)
    if w_elo > 1800 and b_elo > 1800 and abs(w_elo-b_elo) < 400 and 600 <= int(seconds) < 100000 and addition == '0':
        i += 1
        print(f"{i} / {j} skipped-{skipped}")
        new_pgn.write(game.__str__())
        new_pgn.write("\n\n")
    game = read_game(full_pgn)

full_pgn.close()
new_pgn.close()
