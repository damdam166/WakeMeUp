# -*- coding: utf-8 -*-
''' This is the file you have to modify for the tournament. Your default AI player must be called by this module, in the
myPlayer class.

Right now, this class contains the copy of the randomPlayer. But you have to change this!
'''

import time
import Goban 
from playerInterface import *
import numpy as np
import torch
import torch.nn as nn

class GoNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.conv1(x)) # [batch, 32, 8, 8]
        x = torch.relu(self.conv2(x)) # [batch, 64, 8, 8]
        x = x.view(x.size(0), -1) # flatten
        x = torch.relu(self.fc1(x)) # [batch, 128]
        x = self.sigmoid(self.fc2(x)) # [batch, 1]
        return x

class myPlayer(PlayerInterface):
    ''' Example of a random player for the go. The only tricky part is to be able to handle
    the internal representation of moves given by legal_moves() and used by push() and 
    to translate them to the GO-move strings "A1", ..., "J8", "PASS". Easy!

    '''

    def __init__(self):
        self._board = Goban.Board()
        self._mycolor = None
        self.time_left = 30 * 60
        self.model = GoNet()
        self.model.load_state_dict(torch.load("../heuristic/model.pt", weights_only=False))
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')

    def _split_channels(self, matrix: np.ndarray) -> np.ndarray:
        """ Pytorch wants this.
        """
        black: np.ndarray = (matrix == 1).astype(np.float32)
        white: np.ndarray = (matrix == 2).astype(np.float32)
        return np.stack([black, white], axis=0)

    def _transform_position(self, black_stones: list, white_stones: list) -> np.ndarray[int]:
        """
        :return: a numpy array of size (NB_LINES, NB_COLUMNS) such as:
        array[i][j] == 1 if 'str(j)i' belongs to *black*,
        array[i][j] == 2 if 'str(j)i' belongs to *whites*,
        array[i][j] == 0 otherwise.
        """
        # Format input.
        if "PASS" in black_stones:
            black_stones.remove("PASS")
        if "PASS" in white_stones:
            white_stones.remove("PASS")

        matrix = np.zeros(shape=(8, 8), dtype=int)

        # Black stones
        if black_stones != []:
            black_coords: np.ndarray = np.array(black_stones)
            # line is black_coords[:, 1], column is black_coords[:, 0].
            matrix[black_coords[:, 1], black_coords[:, 0]] = 1

        # White stones
        if white_stones != []:
            white_coords: np.ndarray = np.array(white_stones)
            matrix[white_coords[:, 1], white_coords[:, 0]] = 2

        # Torch is waiting for something with the shape=[batch_size, 2, 8, 8].
        return self._split_channels(matrix)

    def _position_predict(self, black_stones: list, white_stones: list, is_white: bool) -> float:
        Xpos_np: np.ndarray = self._transform_position(black_stones, white_stones)
        Xpos: torch.tensor = torch.from_numpy(Xpos_np).float()

        # Add a layer of `[]` to the tensor.
        Xpos = Xpos.unsqueeze(0)

        # Want big gpu.
        Xpos = Xpos.to(self.device)

        prediction = self.model(Xpos)
        result = float(prediction[0][0])
        if is_white:
            result = 1 - result
        return (result - 0.5) * 2

    def _predict_board(self, board: Goban.Board, is_white: bool):
        pieces = [(x,y,board._board[board.flatten((x,y))]) for x in range(board._BOARDSIZE)
                  for y in range(board._BOARDSIZE) if board._board[board.flatten((x,y))] != board._EMPTY]
        return self._position_predict(
            [(x, y) for (x, y, c) in pieces if c == 1],
            [(x, y) for (x, y, c) in pieces if c != 1],
            is_white
        )

    def _sorted_moves(self, board: Goban.Board, is_white: bool, top_m: int = 10):
        """
        :param top_m: this thing is for Monte Carlo, just consider the *top_m*
                                                            best moves.
        """
        move_scores = []
        for move in board.legal_moves():
            board.push(move)
            score: float = -self._predict_board(board, is_white)
            board.pop()
            move_scores.append((score, move))

        move_scores.sort(reverse=True, key=lambda x: x[0])
        return [ move for (_, move) in move_scores[:top_m] ]

    def _alpha_beta(self, board, depth, alpha, beta, is_white):
        if depth == 0 or board.is_game_over():
            return self._predict_board(board, is_white)
        max_eval = float('-inf')
        for move in board.legal_moves():
            board.push(move)
            eval_board = -self._alpha_beta(board, depth - 1, -beta, -alpha, is_white)
            board.pop()
            if eval_board > max_eval:
                max_eval = eval_board
            if max_eval > alpha:
                alpha = max_eval
            if alpha >= beta:
                break  # Beta cut-off
        return max_eval

    def getPlayerName(self):
        return "Wake Me Up"

    def getPlayerMove(self):
        start_time = time.time()

        nb_moves = len(self._board._historyMoveNames)
        number_moves_left = 64 - nb_moves
        epsilon: int = 5
        cool_time: int = min(15, self.time_left / max(1, number_moves_left) - epsilon)

        if self._board.is_game_over():
            print("Referee told me to play but the game is over!")
            return "PASS"

        best_move = None
        depth: int = 1 # Iterative deepening, this will increase.

        # Not to do some crazy stuff at the opening.
        top_m = 10 if nb_moves < 10 else 5

        # Kinda of *Monte Carlo* stuff right here.
        # Just consider the *Candidate* moves like in chess you know.
        moves = self._sorted_moves(self._board, self._mycolor != 1, top_m=top_m)

        alpha = float('-inf')
        beta = float('inf')

        try:
            while True:
                current_best_move = None
                current_best_value = float('-inf')

                # Alpha-beta search for the best move
                for move in moves:
                    self._board.push(move)
                    value = -self._alpha_beta(self._board, depth - 1, -beta, -alpha, self._mycolor != 1)
                    self._board.pop()
                    if value > current_best_value:
                        current_best_value = value
                        current_best_move = move
                    alpha = max(alpha, value)

                best_move = current_best_move
                depth += 1

                if time.time() - start_time > cool_time or depth == 5:
                    raise TimeoutError

            if time.time() - start_time > cool_time or depth == 5:
                raise TimeoutError

            best_move = current_best_move
            depth += 1 # Iterative Deepening, you know.

        except:
            print(f'Iterative Deepening stopped at depth={depth}.')

        self.time_left -= time.time() - start_time
        self._board.push(best_move)
        print("I am playing ", self._board.move_to_str(best_move))
        print("My current board :")
        self._board.prettyPrint()
        return Goban.Board.flat_to_name(best_move)

    def playOpponentMove(self, move):
        print("Opponent played ", move) # New here
        # the board needs an internal represetation to push the move.  Not a string
        self._board.push(Goban.Board.name_to_flat(move)) 

    def newGame(self, color):
        self._mycolor = color
        self._opponent = Goban.Board.flip(color)

    def endGame(self, winner):
        if self._mycolor == winner:
            print("I won!!!")
        else:
            print("I lost :(!!")



