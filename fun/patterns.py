"""Life patterns in RLE (the format of LifeWiki and Golly), and a stamp to put them in a world."""

import re

import numpy as np

# key: (name, RLE body)
PATTERNS = {
    "g": ("Gosper glider gun",
          "24bo$22bobo$12b2o6b2o12b2o$11bo3bo4b2o12b2o$2o8bo5bo3b2o$2o8bo3bob2o4bobo$10bo5bo7bo$"
          "11bo3bo$12b2o!"),
    "n": ("Simkin glider gun",
          "2o5b2o$2o5b2o2$4b2o$4b2o5$22b2ob2o$21bo5bo$21bo6bo2b2o$21b3o3bo3b2o$26bo4$20b2o$20bo$"
          "21b3o$23bo!"),
    "p": ("R-pentomino", "b2o$2o$bo!"),
    "a": ("acorn", "bo$3bo$2o2b3o!"),
    "d": ("diehard", "6bo$2o$bo3b3o!"),
    "l": ("one-line infinite growth", "8ob5o3b3o6b7ob5o!"),
    "w": ("lightweight spaceship", "bo2bo$o$o3bo$4o!"),
    "x": ("pentadecathlon", "2bo4bo$2ob4ob2o$2bo4bo!"),
    "v": ("glider", "bo$2bo$3o!"),
}


def parse(rle):
    """RLE body -> 2D uint8 array. b = dead, o = alive, $ = next row, a number repeats what follows."""
    rows, row = [], []
    for count, tag in re.findall(r"(\d*)([bo$!])", rle):
        n = int(count) if count else 1
        if tag == "b":
            row += [0] * n
        elif tag == "o":
            row += [1] * n
        elif tag == "$":
            rows.append(row)
            rows += [[]] * (n - 1)
            row = []
        else:
            break
    rows.append(row)
    w = max(map(len, rows))
    return np.array([r + [0] * (w - len(r)) for r in rows], dtype=np.uint8)


def pulsar():
    """Period 3 oscillator, 13x13: bars of 3 on rows/columns 0, 5, 7 and 12."""
    p = np.zeros((13, 13), np.uint8)
    for i in (0, 5, 7, 12):
        for j in (2, 8):
            p[i, j:j + 3] = 1
            p[j:j + 3, i] = 1
    return p


SHAPES = {k: (name, parse(rle)) for k, (name, rle) in PATTERNS.items()}
SHAPES["u"] = ("pulsar", pulsar())


def stamp(cells, shape, x, y, turn=0):
    """Put shape (turned by 90 degrees x turn) with its centre at (x, y), wrapping around the edges."""
    s = np.rot90(shape, turn)
    h, w = s.shape
    ys = (np.arange(h) + y - h // 2) % cells.shape[0]
    xs = (np.arange(w) + x - w // 2) % cells.shape[1]
    cells[np.ix_(ys, xs)] = s
