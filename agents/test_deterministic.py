import pytest
import numpy as np

# content of test_sample.py
from agents.deterministic import _possible_alignments, _neighbours_of


def test_alignments_single_ship():
    board = np.zeros((3, 3))
    ships = [2]
    output = _possible_alignments(board, ships)
    expected = np.array([[2, 3, 2],
                         [3, 4, 3],
                         [2, 3, 2]])
    np.testing.assert_array_equal(expected, output)


def test_alignments_single_ship_2():
    board = np.zeros((5, 5))
    ships = [3]
    output = _possible_alignments(board, ships)
    expected = np.array([[2, 3, 4, 3, 2],
                         [3, 4, 5, 4, 3],
                         [4, 5, 6, 5, 4],
                         [3, 4, 5, 4, 3],
                         [2, 3, 4, 3, 2]])
    np.testing.assert_array_equal(expected, output)


def test_alignments_multi_ship():
    board = np.zeros((3, 3))
    ships = [2, 3]
    output = _possible_alignments(board, ships)
    expected = np.array([[4, 5, 4],
                         [5, 6, 5],
                         [4, 5, 4]])
    np.testing.assert_array_equal(expected, output)


def test_alignments_pre_shot_board():
    board = np.array([[1, 0, 0],
                      [0, 0, 1],
                      [0, 0, 0]])
    ships = [2]
    output = _possible_alignments(board, ships)
    expected = np.array([[0, 2, 1],
                         [2, 3, 0],
                         [2, 3, 1]])
    np.testing.assert_array_equal(expected, output)


def test_neighbours_of():
    board = np.array([[0, 1, 0],
                      [2, 1, 1],
                      [0, 2, 0]])
    coordinates = np.array([[1, 0], [2, 1]])
    output = _neighbours_of(board, coordinates)
    expected = np.array([[0, 0], [2, 0], [2, 0], [2, 2]])
    np.testing.assert_array_equal(expected, output)
