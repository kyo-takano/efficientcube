"""
This module contains class implementations for three environments:
- Rubik's Cube: Cube3
- 15 Puzzle: Puzzle15
- Lights Out: LightsOut7

Please note that we prioritize readability and reproducibility over speed optimization in this repository.
"""

import random
import numpy as np

class Cube3:
    """
    A class for 3x3x3 Rubik's Cube
    """
    def __init__(self):
        self.DTYPE = np.int64

        # Define initial and goal state
        self.reset()
        self.goal = np.arange(0, 9 * 6, dtype=self.DTYPE) // 9

        # Define moves
        ## faces and turns
        faces = ["U", "D", "L", "R", "B", "F"]
        ## [90 degrees clockwise, 90 degrees counter-clockwise]
        degrees = ["", "'"]
        degrees_inference = degrees[::-1]
        self.moves = [f"{f}{n}" for f in faces for n in degrees]
        self.moves_inference = [f"{f}{n}" for f in faces for n in degrees_inference]

        # Opposite faces
        self.pairing = {
            "R": "L",
            "L": "R",
            "F": "B",
            "B": "F",
            "U": "D",
            "D": "U",
        }
        # Prohibit obviously redundant moves.
        self.moves_available_after = {
            m: [v for v in self.moves if v[0] != m[0]] + [m] 
            for m in self.moves
        } # self-cancelling moves on the same face

        # [OPTIMIZATION] slicing by move string (e.g., R', U, F) => indices (e.g., 2, 6, 1)
        self.moves_ix = [self.moves.index(m) for m in self.moves]
        self.moves_ix_available_after = {
            self.moves.index(m): [self.moves.index(m) for m in available_moves]
            for m, available_moves in self.moves_available_after.items()
        }
        self.moves_ix_inference = [self.moves.index(m) for m in self.moves_inference]
        self.pairing_ix = {
            0: 1,
            1: 0,
            2: 3,
            3: 2,
            4: 5,
            5: 4,
        } # Points to the opposite face index

        # Vectorize the sticker group replacement operations
        self.__vectorize_moves()

    def reset(self):
        """Resets the cube state to the solved state."""
        self.state = np.arange(0, 9 * 6, dtype=self.DTYPE) // 9

    def is_solved(self):
        """Checks if the cube is in the solved state."""
        return np.all(self.state == self.goal)

    def finger(self, move):
        """Applies a single move on the cube state using move string."""
        self.state[self.sticker_target[move]] = self.state[self.sticker_source[move]]

    def finger_ix(self, ix):
        """The same `finger` method **but using indices of moves for faster execution"""
        self.state[self.sticker_target_ix[ix]] = self.state[self.sticker_source_ix[ix]]

    def apply_scramble(self, scramble):
        """Applies a sequence of moves (scramble) to the cube state."""
        if isinstance(scramble, str):
            scramble = scramble.split()
        for m in scramble:
            if m[-1]=='2':
                for _ in range(2):
                    self.finger(m[0])
            else:
                    self.finger(m)

    def scrambler(self, scramble_length):
        """
        Generates a random scramble of given length and returns the cube state and scramble moves as a generator.
        Please note that index-based implementations (faster) follow commented lexical logics.
        """
        while True:
            # Reset the cube state, scramble, and return cube state and scramble moves
            self.reset()
            scramble = []

            for i in range(scramble_length):
                if i:
                    last_move = scramble[-1]
                    if i > 1:   # [3rd~ moves]
                        while True:
                            # move = random.choice(self.moves_available_after[last_move])
                            move = random.choice(self.moves_ix_available_after[last_move])

                            if scramble[-2] == last_move == move:
                                # Three subsequent moves on the same face, which could be one
                                continue
                            # elif (
                            #     scramble[-2][0] == move[0] and len(scramble[-2] + move) == 3
                            #     and last_move[0] == self.pairing[move[0]]
                            # ):
                            elif (
                                scramble[-2]//2 == move//2 and scramble[-2]%2 != move%2
                                and last_move//2 == self.pairing_ix[move//2]
                            ):
                                # Two mutually canceling moves sandwiching an opposite face move
                                continue
                            else:
                                break
                    else:       # [2nd move]
                        # move = random.choice(self.moves_available_after[last_move])
                        move = random.choice(self.moves_ix_available_after[last_move])
                else:           # [1st move]
                    # move = random.choice(self.moves)
                    move = random.choice(self.moves_ix)

                # self.finger(move)
                self.finger_ix(move)
                scramble.append(move)

                yield self.state, move


    def __vectorize_moves(self):
        """
        Vectorizes the sticker group replacement operations for faster computation.
        This method defines ```self.sticker_target``` and ```self.sticker_source``` to manage sticker colors (target is replaced by source).
        They define indices of target and source stickers so that the moves can be vectorized.

        Colors:

                0 0 0
                0 0 0
                0 0 0
        2 2 2   5 5 5   3 3 3   4 4 4
        2 2 2   5 5 5   3 3 3   4 4 4
        2 2 2   5 5 5   3 3 3   4 4 4
                1 1 1
                1 1 1
                1 1 1

        Order of stickers on each face:

             2   5   8
             1   4   7
            [0]  3   6

        Indices of state (each starting with 9*(n-1)):

                         2   5   8
                         1   4   7
                        [0]  3   6
             20  23 26  47  50  53  29  32 35  38  41 44
             19  22 25  46  49  52  28  31 34  37  40 43
            [18] 21 24 [45] 48  51 [27] 30 33 [36] 39 42
                        11   14 17
                        10   13 16
                        [9]  12 15
        """
        self.sticker_target, self.sticker_source = dict(), dict()

        self.sticker_replacement = {
            # Sticker A is replaced by another sticker at index B -> A:B
            'U':{0: 6, 1: 3, 2: 0, 3: 7, 5: 1, 6: 8, 7: 5, 8: 2, 20: 47, 23: 50, 26: 53, 29: 38, 32: 41, 35: 44, 38: 20, 41: 23, 44: 26, 47: 29, 50: 32, 53: 35},
            'D':{9: 15, 10: 12, 11: 9, 12: 16, 14: 10, 15: 17, 16: 14, 17: 11, 18: 36, 21: 39, 24: 42, 27: 45, 30: 48, 33: 51, 36: 27, 39: 30, 42: 33, 45: 18, 48: 21, 51: 24},
            'L':{0: 44, 1: 43, 2: 42, 9: 45, 10: 46, 11: 47, 18: 24, 19: 21, 20: 18, 21: 25, 23: 19, 24: 26, 25: 23, 26: 20, 42: 11, 43: 10, 44: 9, 45: 0, 46: 1, 47: 2},
            'R':{6: 51, 7: 52, 8: 53, 15: 38, 16: 37, 17: 36, 27: 33, 28: 30, 29: 27, 30: 34, 32: 28, 33: 35, 34: 32, 35: 29, 36: 8, 37: 7, 38: 6, 51: 15, 52: 16, 53: 17},
            'B':{2: 35, 5: 34, 8: 33, 9: 20, 12: 19, 15: 18, 18: 2, 19: 5, 20: 8, 33: 9, 34: 12, 35: 15, 36: 42, 37: 39, 38: 36, 39: 43, 41: 37, 42: 44, 43: 41, 44: 38},
            'F':{0: 24, 3: 25, 6: 26, 11: 27, 14: 28, 17: 29, 24: 17, 25: 14, 26: 11, 27: 6, 28: 3, 29: 0, 45: 51, 46: 48, 47: 45, 48: 52, 50: 46, 51: 53, 52: 50, 53: 47}
        }
        for m in self.moves:
            if len(m) == 1:
                assert m in self.sticker_replacement
            else:
                if "'" in m:
                    self.sticker_replacement[m] = {
                        v: k for k, v in self.sticker_replacement[m[0]].items()
                    }
                elif "2" in m:
                    self.sticker_replacement[m] = {
                        k: self.sticker_replacement[m[0]][v]
                        for k, v in self.sticker_replacement[m[0]].items()
                    }
                else:
                    raise

            self.sticker_target[m] = list(self.sticker_replacement[m].keys())
            self.sticker_source[m] = list(self.sticker_replacement[m].values())

            for i, idx in enumerate(self.sticker_target[m]):
                assert self.sticker_replacement[m][idx] == self.sticker_source[m][i]

        # For index slicing
        self.sticker_target_ix = np.array([np.array(self.sticker_target[m]) for m in self.moves])
        self.sticker_source_ix = np.array([np.array(self.sticker_source[m]) for m in self.moves])

class Puzzle15:
    def __init__(self):
        self.DTYPE = np.int64

        # define state and goal
        self.reset() # state
        self.goal = np.concatenate((np.arange(1, 4 * 4, dtype=self.DTYPE), [0]))

        self.moves = ['U', 'D', 'L', 'R']
        self.moves_inference = ['D','U','R','L']

        # opposite faces
        self.pairing = {
            'R':'L',
            'L':'R',
            'U':'D',
            'D':'U',
        }
        self.moves_available_after = {
            m:[v for v in self.moves if v!=self.pairing[m]] for m in self.moves
        }

        # [OPTIMIZATION] slicing by move string => indices
        self.moves_ix = [self.moves.index(m) for m in self.moves]
        self.moves_ix_available_after = {
            self.moves.index(m): [self.moves.index(m) for m in available_moves]
            for m, available_moves in self.moves_available_after.items()
        }
        self.moves_ix_inference = [self.moves.index(m) for m in self.moves_inference]
        self.pairing_ix = {
            0:1,
            1:0,
            2:3,
            3:2
        }

        # vectorize the sticker group replacement operations
        self.__vectorize_moves()

    def reset(self):
        self.state = np.concatenate((np.arange(1, 4 * 4, dtype=self.DTYPE), [0]))

    def is_solved(self):
        return np.all(self.state == self.goal)

    def finger(self, move):
        if isinstance(move, str):
            move = self.moves.index(move)
        self.finger_ix(move)

    def finger_ix(self, move):
        # [target] empty slot
        target_index = np.squeeze(np.where(self.state == 0))
        # [source] to-be empty slot
        source_index = self.swap_zeros[target_index, move] # `self.swap_zeros`: defined in `__vectorize_moves`
        # Swap.
        self.state[target_index], self.state[source_index] = self.state[source_index], 0

    def apply_scramble(self, scramble):
        if isinstance(scramble, str):
            scramble = scramble.split()
        for m in scramble:
            self.finger(m)

    def scrambler(self, scramble_length):
        """
            A generator function yielding the state and scramble
        """
        while True:
            # reset the self.state, scramble, and retun self.state and scramble moves
            self.reset()
            scramble = []
            for i in range(scramble_length):
                target_loc = np.where(self.state.reshape(4, 4) == 0)
                for _ in iter(int,1):
                    if scramble:    # [2nd~ move]
                        move = random.choice(self.moves_ix_available_after[scramble[-1]])
                    else:           # [1st move]
                        move = random.choice(self.moves_ix)
                    """
                    index_map:
                        [ 0  1  2  3]
                        [ 4  5  6  7]
                        [ 8  9 10 11]
                        [12 13 14 15]
                    """
                    # Skip ineffective moves
                    if move==3: # if move=="R"
                        if target_loc[1]:
                            # zero_index NOT on the left
                            break
                    elif move==1: # if move=="D"
                        if target_loc[0]:
                            # zero_index NOT be at the top
                            break
                    elif move==0: # if move=="U"
                        if target_loc[0]!=4-1:
                            # zero_index NOT on the left
                            break
                    elif move==2: # if move=="L"
                        if target_loc[1]!=4-1:
                            # zero_index NOT on the left
                            break
                    else:
                        raise ValueError("Unexpected move index.")

                self.finger_ix(move)
                scramble.append(move)
                yield self.state, move

    def __vectorize_moves(self):
        # Code retrieved & modified from:
        # https://github.com/forestagostinelli/DeepCubeA/blob/f75918a5fb83c140b649ee634f765d42845b17a3/environments/n_puzzle.py#L174
        self.swap_zeros = np.zeros((4*4, len(self.moves)), dtype=self.DTYPE)
        for move_ix, move in enumerate(self.moves):
            for i in range(4):
                for j in range(4):
                    z_idx = np.ravel_multi_index((i, j), (4, 4))
                    state = np.ones((4, 4), dtype=np.int64)
                    state[i, j] = 0

                    is_eligible = False
                    if move == 'U':
                        is_eligible = i < (4 - 1)
                    elif move == 'D':
                        is_eligible = i > 0
                    elif move == 'L':
                        is_eligible = j < (4 - 1)
                    elif move == 'R':
                        is_eligible = j > 0

                    if is_eligible:
                        swap_i, swap_j = -1, -1
                        if move == 'U':
                            swap_i = i + 1
                            swap_j = j
                        elif move == 'D':
                            swap_i = i - 1
                            swap_j = j
                        elif move == 'L':
                            swap_i = i
                            swap_j = j + 1
                        elif move == 'R':
                            swap_i = i
                            swap_j = j - 1

                        self.swap_zeros[z_idx, move_ix] = np.ravel_multi_index((swap_i, swap_j), (4, 4))
                    else:
                        self.swap_zeros[z_idx, move_ix] = z_idx

class LightsOut7:
    def __init__(self, dim=7):
        self.dtype = np.int64

        # define state and goal
        self.dim = dim
        self.num_tiles = self.dim ** 2
        self.moves = list(range(self.num_tiles))
        self.moves_inference = self.moves
        self.reset()

        # To be consistent with the other environments:
        self.moves_ix_inference = self.moves_inference
        # vectorize the sticker group replacement operations
        self.__vectorize_moves()

    def reset(self):
        self.state = np.zeros((self.num_tiles), dtype=self.dtype)

    def is_solved(self):
        return np.all(self.state == 0)

    def finger(self, move):
        self.state[self.move_matrix[move]] = (self.state[self.move_matrix[move]] + 1) % 2 # Take modulo of odd/even to simulate boolean

    def finger_ix(self, move):
        self.finger(move)

    def apply_scramble(self, scramble):
        if isinstance(scramble, str):
            scramble = scramble.split()
        try:
            scramble = [int(m) for m in scramble]
        except:
            raise TypeError('Scramble sequence to **Lights Out** must be either a list of integers or a *space-delimited* list of integers')

        for m in scramble:
            self.finger(m)

    def scrambler(self, scramble_length=80):
        while True:
            self.reset()
            scramble = []
            moves = list(np.random.permutation(scramble_length))
            for i in range(scramble_length):
                move = moves[i]
                self.finger(move)
                scramble.append(move)

                # yield self.state, move
                yield self.state, scramble

    def __vectorize_moves(self):
        self.move_matrix = np.zeros((self.num_tiles, 5), dtype=np.int64)
        for move in range(self.num_tiles):
            x_pos = int(np.floor(move / self.dim))
            y_pos = move % self.dim

            right = move + self.dim if x_pos < (self.dim-1) else move
            left = move - self.dim if x_pos > 0 else move
            up = move + 1 if y_pos < (self.dim - 1) else move
            down = move - 1 if y_pos > 0 else move

            self.move_matrix[move] = [move, right, left, up, down]

def load_environment(name: str, verbose=False):
    # Unify notation
    name = name.strip()
    name_unified = ''.join(name.lower().split())
    # Find the corresponding problem
    if name_unified in ["rubik'scube", "rubixcube", "cube3", "cube3x3", "cube3x3x3"]:
        return Cube3()
    elif name_unified in ["15puzzle", "puzzle15"]:
        return Puzzle15()
    elif name_unified in ["lightsout", "lightsout7", "lightsout7x7", "7x7lightsout"]:
        return LightsOut7()
    else:
        # No correspondence.
        # Find & suggest the lexically nearest option
        # Code retrieved & modified from https://gist.github.com/kyo-takano/fa2b42fb4df20e2566c29c31f20f87ed
        import gzip
        query = name
        Q = gzip.compress(query.encode())
        distance_from_Q = {}
        for chunk in ["Rubik's Cube", "15 Puzzle", "Lights Out"]:
            C = gzip.compress(chunk.encode())
            query_chunk = query + " " + chunk
            Q_C = gzip.compress(query_chunk.encode())
            normalized_distance = (len(Q_C) - min(len(Q), len(C))) / max(len(Q), len(C))
            distance_from_Q[chunk] = normalized_distance
        nearest = sorted(distance_from_Q, key=distance_from_Q.get)[0]
        if verbose:
            print(f"Distance: {distance_from_Q[nearest]}")
        raise ValueError(f'Invalid environment name. Did you mean: "{nearest}"?')

if __name__=="__main__":
    env = Cube3()
    print(f"Goal:\n{env.state.reshape(6,9)}")
    env.apply_scramble("R R F U' B F D L D R'")
    print(f"Scrambled:\n{env.state.reshape(6,9)}")
