from chess.pgn import read_game
from timeit import default_timer as timer
from datetime import timedelta


full_pgn = open("lichess_db_standard_rated_2017-04.pgn")
new_pgn = open("classic_600+0_200_1800.pgn", "w")

start = timer()
game = read_game(full_pgn)
i, j = 0, 0
while game is not None:
    j += 1
    headers = game.headers
    event = headers["Event"]
    w_elo = int(headers["WhiteElo"])
    b_elo = int(headers["BlackElo"])
    time_control = headers["TimeControl"]
    if event == "Rated Classical game" and w_elo > 1800 and b_elo > 1800 and abs(w_elo-b_elo) > 200 and time_control == "600+0":
        i += 1
        print(f"{i} / {j}")
        new_pgn.write(game.__str__())
        new_pgn.write("\n\n")
    game = read_game(full_pgn)

end = timer()
print(timedelta(seconds=end-start))
print(end - start)

full_pgn.close()
new_pgn.close()
