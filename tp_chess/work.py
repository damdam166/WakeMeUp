# ---------------------------------------------------------------------------
# Some functions to get results about the `Chess` game.
# ---------------------------------------------------------------------------

import time
import chess

INFINITE_SCORE = 1000000

# ---------------------------------------------------------------------------
# How to find the number of possible games in `Chess`.
# ---------------------------------------------------------------------------

def find_chess_games(board: chess.Board, depth: int = 2) -> int:
    """
    :return: The number of explored nodes.
    """
    if (board.is_game_over() or depth == 0):
        return 1

    number_current_nodes: int = 0
    for move in board.generate_legal_moves():
        board.push(move)
        number_current_nodes += 1 + find_chess_games(board, depth - 1)
        current_move: chess.Move = board.pop()

    return number_current_nodes

# Display results.
#for depth in range(1, 10):
    #start = time.time()
    #nb_explored_nodes: int = find_chess_games(chess.Board(), depth = depth)
    #end = time.time()
#
    #time_cost: int = round(end - start, 3) # seconds
#
    #print(f'The number of explored nodes for chess games using depth={depth} is : \
          #{nb_explored_nodes}. \n \
          #It took {time_cost} seconds. \n \
          #')

# ---------------------------------------------------------------------------
# `Chess` heuristic from *Claude Shannon*, **1950**.
# ---------------------------------------------------------------------------

def chess_result(board: chess.Board) -> int:
    """
    !!! The game should be over when calling this function. !!!
    """
    result: str = board.result()

    if result == "1-0":
        return +INFINITE_SCORE # `chess.WHITE` won
    elif result == "0-1":
        return -INFINITE_SCORE # `chess.BLACK` won
    return 0

def chess_heuristic(board: chess.Board) -> int:
    """
    :param a chess board that could be finished or not.
    :return: It follow this scheme:

        If game finished --> returns the game result:
                                        - 0 if tie.
                                        - +INFINITE_SCORE if whites win.
                                        - -INFINITE_SCORE if blacks win.

        else:
           count the piece values such as:
            - *pawn* = 1
            - *knight* = 3
            - *bishop* = 3
            - *rook* = 5
            - *queen* = 9

            returns *score_whites* - **score_blacks*.
    """
    if board.is_game_over():
        return chess_result(board)

    # Count pieces.
    score: int = 0

    pieces: dict[int, chess.Piece] = board.piece_map() # `key` is square_index.
    for piece in pieces.values():
        # Gets the symbol ``P``, ``N``, ``B``, ``R``, ``Q`` or ``K`` for white
        # pieces or the lower-case variants for the black pieces.
        symbol: str = piece.symbol()

        if symbol == 'Q': # Whites
            score += 9
        elif symbol == 'R':
            score += 5
        elif symbol == 'B':
            score += 3
        elif symbol == 'N':
            score += 3
        elif symbol == 'P':
            score += 1
        elif symbol == 'q': # Blacks
            score -= 9
        elif symbol == 'r':
            score -= 5
        elif symbol == 'b':
            score -= 3
        elif symbol == 'n':
            score -= 3
        elif symbol == 'p':
            score -= 1

        # Do nothing, do not valuate the king.

    return score

#print(f'Score for the basic *chess* position: \
      #{chess_heuristic(chess.Board())} \
      #')
#
#board_test_2: chess.Board = chess.Board(
    #"r1bqkb1r/pppp1ppp/2B2n2/4p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 0 4")
#
#print(f'Score for a +3 *chess* position: \
      #{chess_heuristic(board_test_2)} \
      #')

# ---------------------------------------------------------------------------
# Minimax version of *Chess*.
# ---------------------------------------------------------------------------

def minimax_chess(board: chess.Board, depth: int = 4,
                  wantMax: bool = True) -> (int, list):
    """
    :param depth: What is the last layer before calculating the heuristic?
    :param wantMax: True --> find score that is good for *Whites*.
    :return: (score, `list_moves`). `list_moves` need to be reversed.
    """
    if (depth <= 0 or board.is_game_over()):
        return ( chess_heuristic(board), [] )

    if wantMax:
        score: int = -INFINITE_SCORE
        list_moves: list = []
        for move in board.generate_legal_moves():
            board.push(move)
            (current_score, list_previous_moves) = \
                minimax_chess(board, depth - 1, False)
            current_move: chess.Board = board.pop() # == move

            score = max(score, current_score)
            if score <= current_score:
                list_moves = list_previous_moves.copy()
                list_moves.append(move.uci())

        return ( score, list_moves )

    else:
        score: int = +INFINITE_SCORE
        list_moves: list = []
        for move in board.generate_legal_moves():
            board.push(move)
            (current_score, list_previous_moves) = \
                minimax_chess(board, depth - 1, True)
            current_move: chess.Board = board.pop() # == move

            score = min(score, current_score)
            if current_score <= score:
                list_moves = list_previous_moves.copy()
                list_moves.append(move.uci())

        return ( score, list_moves )

#board_test_1: chess.Board = chess.Board()
#depth: int = 4

#start: int = time.time()
#(score_1, list_moves_1) = minimax_chess(board_test_1, depth=depth)
#end: int = time.time()
#time_cost: int = round(end - start, 3) # in seconds.

#list_moves_1.reverse()
#for move in list_moves_1:
    #board_test_1.push(chess.Move.from_uci(move))

#print(f'Testing minimax for *Chess*. \n \
      #Basic start with depth = {depth} gives, in {time_cost} seconds: \n \
          #score = {score_1} \n \
          #moves are: \n \
#{list_moves_1} \n \
#{list_moves_1} \n \
#\n \
      #The current game position is: \n \
#{board_test_1.fen()} \n \
#\n \
      #Here is the current board: \n \
#{board_test_1.unicode()} \n \
#')

# ---------------------------------------------------------------------------
# AlphaBeta version of *Chess*.
# ---------------------------------------------------------------------------

def alphabeta_chess(board: chess.Board, depth: int = 4,
                    alpha: int = -INFINITE_SCORE, beta: int = +INFINITE_SCORE,
                    wantMax: bool = True) -> (int, list):
    """
    :param depth: What is the last layer before calculating the heuristic?
    :param wantMax: True --> find score that is good for *Whites*.
    :return: (score, `list_moves`). `list_moves` need to be reversed.
    """
    if (depth <= 0 or board.is_game_over()):
        return ( chess_heuristic(board), [] )

    if wantMax:
        score: int = -INFINITE_SCORE
        list_moves: list = []
        for move in board.generate_legal_moves():
            board.push(move)
            (current_score, list_previous_moves) = \
                alphabeta_chess(board, depth - 1, alpha, beta, False)
            current_move: chess.Board = board.pop() # == move

            score = max(score, current_score)
            if score <= current_score:
                list_moves = list_previous_moves.copy()
                list_moves.append(move.uci())

            alpha = max(alpha, score)
            if beta <= alpha:
                break

        return ( score, list_moves )

    else:
        score: int = +INFINITE_SCORE
        list_moves: list = []
        for move in board.generate_legal_moves():
            board.push(move)
            (current_score, list_previous_moves) = \
                alphabeta_chess(board, depth - 1, alpha, beta, True)
            current_move: chess.Board = board.pop() # == move

            score = min(score, current_score)
            if current_score <= score:
                list_moves = list_previous_moves.copy()
                list_moves.append(move.uci())

            beta = min(beta, score)
            if beta <= alpha:
                break

        return ( score, list_moves )

#board_test_1: chess.Board = chess.Board()
#depth: int = 6
#
#start: int = time.time()
#(score_1, list_moves_1) = alphabeta_chess(board_test_1, depth=depth)
#end: int = time.time()
#time_cost: int = round(end - start, 3) # in seconds.
#
#list_moves_1.reverse()
#for move in list_moves_1:
    #board_test_1.push(chess.Move.from_uci(move))
#
#print(f'Testing alphabeta for *Chess*. \n \
      #Basic start with depth = {depth} gives, in {time_cost} seconds: \n \
          #score = {score_1} \n \
          #moves are: \n \
#{list_moves_1} \n \
#\n \
      #The current game position is: \n \
#{board_test_1.fen()} \n \
#\n \
      #Here is the current board: \n \
#{board_test_1.unicode()} \n \
#')

# ---------------------------------------------------------------------------
# AlphaBeta version of *Chess*, with *iterative deepening*.
# ---------------------------------------------------------------------------

lastTimestamp: list[float] = []
def outOfTime() -> bool:
    """
    :return: It measures timestamps. Returns `True` if time is over.
    """
    maxTime: float = 120 # Seconds.
    time.sleep(1)
    lastTimestamp.append(time.time())

    if len(lastTimestamp) <= 1:
        return ( lastTimestamp[0] - maxTime ) <= 0

    currentTimeUsed: float = lastTimestamp[-2] - lastTimestamp[-1]
    return ( maxTime - currentTimeUsed ) <= 0

def iterative_deepening_chess(board: chess.Board, maxDepth: int = 4,
                    wantMax: bool = True) -> str:
    """
    :param maxDepth: a POSITIVE integer. Otherwise it will never stops.
                                It goes from 1 to *maxDepth* included.
    :return: The *iterative deepening* version of alpha beta.
                Just the move in his *uci* form.
    """
    bestMoveToReturn: str = None
    currentPosition: chess.Board = board.copy()
    depth: int = 1

    while (depth <= maxDepth and not outOfTime()):
        currentPosition: chess.Board = board.copy()

        start: int = time.time()
        (score, list_moves) = alphabeta_chess(currentPosition, depth=depth,
                                                        wantMax=wantMax)
        end: int = time.time()
        time_cost: int = round(end - start, 3) # in seconds.

        list_moves.reverse()
        for move in list_moves:
            currentPosition.push(chess.Move.from_uci(move))

        print(f'Testing alphabeta for *Chess*. \n \
                Depth = {depth} gives, in {time_cost} seconds: \n \
                  score = {score} \n \
                  moves are: \n \
        {list_moves} \n \
        \n \
              The current game position is: \n \
        {currentPosition.fen()} \n \
        \n \
              Here is the current board: \n \
{currentPosition.unicode()} \n \
        ')

        # Iteration
        bestMoveToReturn = list_moves[0]
        depth += 1

    return bestMoveToReturn

board_test_1: chess.Board = chess.Board()
maxDepth: int = 10
iterative_deepening_chess(board_test_1, maxDepth=maxDepth, wantMax=True)

# ---------------------------------------------------------------------------

