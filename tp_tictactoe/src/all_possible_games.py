# ---------------------------------------------------------------------------
# Some functions to get results about the `Tictactoe` game.
# ---------------------------------------------------------------------------

import time
import numpy as np
from Tictactoe import Board

INFINITE_SCORE: int = 10000
INFINITE_DEPTH: int = 10000

# ---------------------------------------------------------------------------
# How to find the number of possible games in `Tictactoe`.
# ---------------------------------------------------------------------------

def find_number_games(board: Board, result: int = 0) -> int:
    if (board.is_game_over()):
        return 1

    for move in board.legal_moves():
        board.push(move)
        result += find_number_games(board)
        board.pop()

    return result

#print(f'Number of possible games: {find_number_games(Board())}')

# ---------------------------------------------------------------------------
# How many node are in the `Tictactoe` tree.
# ---------------------------------------------------------------------------

def find_number_nodes(board: Board, result: int = 0) -> int:
    if (board.is_game_over()):
        return 1

    for move in board.legal_moves():
        board.push(move)
        result += find_number_nodes(board)
        board.pop()

    return result + 1

#print(f'Number of nodes in the tree: {find_number_nodes(Board())}')

# ---------------------------------------------------------------------------
# To know the time need to explore all the possible games.
# ---------------------------------------------------------------------------

def find_time_explore_tictactoe(board: Board, result: int = 0) -> int:
    if (board.is_game_over()):
        return 0

    for move in board.legal_moves():
        board.push(move)
        result += find_number_nodes(board)
        board.pop()

    return result

start: int = time.time()
find_time_explore_tictactoe(Board())
stop: int = time.time()

#print(f'Time needed to explore all games: {stop - start}seconds')

# ---------------------------------------------------------------------------
# To get a winning strategy without any *shared horizon*.
# ---------------------------------------------------------------------------

def static_evaluation_minimax(board: Board) -> int:
    """
    :return a static evaluation.
            It is not needed without any *shared horizon*.
    """
    if board.result() == 'X':
        return +INFINITE_SCORE
    elif board.result() == 'O':
        return -INFINITE_SCORE

    static_score: int = len(board.legal_moves())

    for move in board.legal_moves():
        board.push(move)

        if board.is_game_over() and board.result() == 'X':
            static_score += 10
        elif board.is_game_over() and board.result() == 'O':
            static_score -= 10

        board.pop()

    return static_score

def maximin(board: Board, depth: int = 3, wantMax: bool = True) -> (int, list):
    """ To find a winning strategy for `X` in a *Tictactoe* game,
                                                    using *MinMax*.

    :param depth: without any *shared horizon*, depth would be infinite.
                    The game stops till someone won, or it is tie.

    :param wantMax: True --> want to take the move that maximizes score,
                                            for the current position.
                    False --> same but the minimum score is considered.

    :return: (score: int, list_previous_moves: list).
    `list_previous_moves` is created by the move
                            who maximizes(wantMax=True) score.
    """
    if (depth == 0 or board.is_game_over()):
        return ( static_evaluation_minimax(board), [] )

    # To return
    score: int = 0
    list_moves_from_chosen_move: list = []

    # Keys are `tuple(move)`.
    # values are (score: int, list_previous_moves: list)
    dict_list_moves: dict = {}

    for move in board.legal_moves():
        board.push(move)

        newWantMax: bool = False if wantMax == True else True
        dict_list_moves[tuple(move)] = maximin(board, depth - 1, newWantMax)

        board.pop()

    if wantMax:
        # Find the move that maximizes score.
        argmax: tuple = max(
            dict_list_moves,
            key = { k:v[0] for k,v in dict_list_moves.items() }.get
        )

        score = dict_list_moves[argmax][0]
        list_moves_from_chosen_move = dict_list_moves[argmax][1].copy()

        # Add the move, need to reconstruct it.
        list_moves_from_chosen_move.append(list(argmax))

    else:
        # Find the move that minimizes score.
        argmin: tuple = min(
            dict_list_moves,
            key = { k:v[0] for k,v in dict_list_moves.items() }.get
        )

        score = dict_list_moves[argmin][0]
        list_moves_from_chosen_move = dict_list_moves[argmin][1].copy()

        # Add the move, need to reconstruct it.
        list_moves_from_chosen_move.append(list(argmin))

    return (score, list_moves_from_chosen_move)

#print(f' \
    #About the score:\n \
    #\'X\'   Winner --> +1\n \
    #\'O\'   Winner --> -1\n \
     #None Winner --> 0\n \
#')
#
#(score, L) = maximin(Board(), depth=+INFINITE_DEPTH)
#L.reverse()
#print(f' \n \
      #Maximin for \'X\' gives the score: {score}. \n \
      #The moves are: \n \
#{L} \n \
#')

# ---------------------------------------------------------------------------
# To get a winning strategy with *shared horizon*, *alpha beta* version.
# ---------------------------------------------------------------------------

def alpha_beta(board: Board,
               depth: int = INFINITE_DEPTH,
               alpha: int = -INFINITE_SCORE, beta: int = +INFINITE_SCORE,
               wantMax: bool = True) -> (int, list):
    """ To find a winning strategy for `X` in a *Tictactoe* game,
                                                    using *AlphaBeta*.

    :param wantMax: True --> want to take the move that maximizes score,
                                            for the current position.
                    False --> same but the minimum score is considered.

    :return: (score: int, list_previous_moves: list).
    `list_previous_moves` is created by the move
                            who maximizes(wantMax=True) score.
    """
    if (depth == 0 or board.is_game_over()):
        return ( static_evaluation_minimax(board), [] )

    if wantMax:
        score: int = -INFINITE_SCORE
        list_moves: list = []
        for move in board.legal_moves():
            board.push(move)
            (current_score, list_previous_moves) = \
                alpha_beta(board, depth - 1, alpha, beta, False)
            board.pop()

            if current_score > score:
                score = current_score
                list_moves = list_previous_moves.copy()
                list_moves.append(move)

            alpha: int = max(alpha, score)
            if beta <= alpha:
                break

        return (score, list_moves)

    else:
        score: int = +INFINITE_SCORE
        list_moves: list = []
        for move in board.legal_moves():
            board.push(move)
            (current_score, list_previous_moves) = \
                alpha_beta(board, depth - 1, alpha, beta, True)
            board.pop()

            if current_score < score:
                score = current_score
                list_moves = list_previous_moves.copy()
                list_moves.append(move)

            beta: int = min(beta, score)
            if beta <= alpha:
                break

        return (score, list_moves)

print(f' \
    About the score:\n \
    \'X\'   Winner --> +{INFINITE_SCORE}\n \
    \'O\'   Winner --> -{INFINITE_SCORE}\n \
     None Winner --> 0\n \
')

(score, L) = alpha_beta(Board(), depth=+INFINITE_SCORE)
L.reverse()
print(f' \n \
      AlphaBeta for \'X\' gives the score: {score}. \n \
      The moves are: \n \
{L} \n \
')

# ---------------------------------------------------------------------------

