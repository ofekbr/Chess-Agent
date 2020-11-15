import chess.pgn
import numpy as np

inputgames = []
inputtimes = []
inputmoves = []


with open("test_games.pgn") as pgn:
    game=chess.pgn.read_game(pgn)
    while game:
        board=game.board()

        for move in game.mainline_moves():
            inputgames.append(board.__str__())
            inputmoves.append(move.__str__())
            board.push(move)

        for node in game.mainline():
            if node.next() and node.next().next():
                inputtimes.append(node.clock()-node.next().next().clock())

        inputgames = inputgames[:-2]
        inputmoves = inputmoves[:-2]

        game = chess.pgn.read_game(pgn)

print(len(inputmoves))
print(len(inputgames))
print(len(inputtimes))

inputList=[]
for i in range(len(inputmoves)):
    inputList.append([inputgames[i],inputmoves[i],inputtimes[i]])

arr=np.array(inputList);




