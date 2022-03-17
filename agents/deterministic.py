from typing import List

import numpy as np


def _alignments_in(row: int, col: int, board: np.ndarray, ships: List[int]) -> set[tuple[int, int, int, str, int]]:
    """
    Determines which ships can be placed, so they intersect with the given (row,col) coordinate. Then returns a set
     of those ships.
    :param row: row index of the coordinate
    :param col: column index of the coordinate
    :param board: a 2D numpy array containing a float representation of the board.
    :param ships: a list of ints, where each int is the length of a ship on the board.
    :return: A set of tuples, where each tuple has the form: (y_ship, x_ship, ship_length, ship_id)
    """
    valid_alignments = set()  # Set of tuples, where each tuple is a unique ship.
    row_len, col_len = board.shape
    for ship_id, ship_length in enumerate(ships):
        # This index shifts the ship across the coordinate.
        for i in range(0, ship_length):
            # Vertical alignment attempts
            if row - i >= 0 \
                    and row - i + ship_length <= row_len \
                    and np.all(board[row - i:row - i + ship_length, col] == 1):
                # Add a tuple of the ship to identify it.
                valid_alignments.add((row - i, col, ship_length, 'V', ship_id))
            # Horizontal alignment attempts
            if col - i >= 0 \
                    and col - i + ship_length <= col_len \
                    and np.all(board[row, col - i:col - i + ship_length] == 1):
                # Add a tuple of the ship to identify it.
                valid_alignments.add((row, col - i, ship_length, 'H', ship_id))

    return valid_alignments


def _possible_alignments(board: np.ndarray, ships: List[int]) -> np.ndarray:
    """
    Counts how many possible ship alignments there are on each cell on the board. So if for coordinate (0,1), there are
    2 possible ships that can be fit to lie on this coordinate then returned_array[0,1] = 2.
    :param board: a 2D numpy array containing an int representation of the board.
    :param ships: a list of ints, where each int is the length of a ship on the board.
    :return: a 2D numpy array containing integers, each a count of the possible alignments per cell.
    """

    row_len, col_len = board.shape
    # board in which to store the alignments.
    alignments = np.zeros((row_len, col_len), dtype=int)

    # r is the row, c is the column
    for r in range(0, row_len):
        for c in range(0, col_len):
            # only bother with empty cells (as it is otherwise always 0).
            if board[r, c] == 1:
                # find alignments per cell
                ship_alignments = _alignments_in(r, c, board, ships)
                alignments[r, c] = len(ship_alignments)

    return alignments


def _neighbours_of(board: np.ndarray, coords: np.ndarray) -> np.ndarray:
    """
    Find coordinates non-hit neighbours of the hit ships
    :param board: board of the game
    :param coords: array of coordinates of hit ships
    :return: array of coordinates of neighbours.
    """
    valid_neighbours = []
    for (r, c) in coords:
        # above
        if r - 1 >= 0 and board[r - 1, c] == 1:
            valid_neighbours.append((r - 1, c))
        # below
        if r + 1 < board.shape[0] and board[r + 1, c] == 1:
            valid_neighbours.append((r + 1, c))
        # left
        if c - 1 >= 0 and board[r, c - 1] == 1:
            valid_neighbours.append((r, c - 1))
        # right
        if c + 1 < board.shape[1] and board[r, c + 1] == 1:
            valid_neighbours.append((r, c + 1))

    return np.array(valid_neighbours)


# TODO update to work on tensor-based data
def deterministic_policy(board: np.ndarray, remaining_ships: List[int]) -> tuple[int, int]:
    """
    Targets the coordinate with the highest number of possible ship alignments.
    If a hit ship is found, target the neighbouring coordinate with the highest number of alignments.
    :param board: a 2D numpy array containing an int representation of the board.
    :param remaining_ships: a list of ints, where each int is the length of a ship on the board.
    :return: next position (row,col) to shoot on the board
    """

    alignments = _possible_alignments(board, remaining_ships)
    # check if there are unsunk ships visible
    if np.any(board == 2):
        hit_ships = np.argwhere(board == 2)
        valid_neighbours = _neighbours_of(board, hit_ships)
        best = -1
        best_idx = -1
        for (r, c) in valid_neighbours:
            if alignments[r, c] > best:
                best_idx = (r, c)
                best = alignments[r, c]
        return best_idx
    # otherwise target highest alignment
    else:
        return np.unravel_index(np.argmax(alignments), alignments.shape)
