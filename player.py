from board import Direction, Rotation, Action
import math


BEST_WEIGHTS = {
    "w1": [
        [
            0.3364990965510476,
            0.5522940051410093,
            -1.2189343101745362,
            -3.7224030497588005,
            4.337058818100866,
            3.9656182531448008,
            -1.3614930832436811,
            5.2749400008891705,
            -0.9068576119554229,
            -1.3535469736112868
        ],
        [
            -1.5950617717723214,
            1.7219088762973271,
            -1.0405432447404537,
            0.027453371198406185,
            0.9030244280858368,
            -1.5100589185279616,
            2.5022234032717905,
            -0.9800901765684162,
            -0.4763882421577189,
            0.09274268244303485
        ],
        [
            2.2613420301419436,
            2.208392163863593,
            2.2189965304288233,
            -1.2137341212435444,
            -1.228259354446418,
            -1.921042468522276,
            -3.7711289122792575,
            2.8264488175536515,
            -0.1648706236644948,
            1.3378412459402025
        ],
        [
            0.6782459624945325,
            -0.4199800125128328,
            -0.0939551285292689,
            0.8457735146704053,
            1.9211666978498363,
            -0.9073282746718755,
            -2.5064108886519088,
            -0.8579058605140035,
            -0.33119286435499573,
            0.9503216642502836
        ],
        [
            -3.6598851664923826,
            5.721867578534918,
            -1.1504077806734236,
            -1.7652028378228632,
            -1.955961193992914,
            -0.506594296063658,
            0.7898281269889548,
            -1.6691377154030111,
            1.6404259397406753,
            0.7805371602183208
        ]
    ],
    "b1": [
        -0.30683249383029954,
        0.051363289086821196,
        -0.20895221978555878,
        -0.5519768099736879,
        -1.9603743844621255,
        -1.2449430822202219,
        1.0028180789732764,
        -0.07079116840567823,
        -1.4898365871854193,
        -3.482437230964977
    ],
    "w2": [
        [
            -0.2732564649675574
        ],
        [
            -3.150178965736796
        ],
        [
            2.3408850626726676
        ],
        [
            -1.29684690880674
        ],
        [
            -1.4070248758821795
        ],
        [
            -3.5154733551655912
        ],
        [
            1.219343501017138
        ],
        [
            -1.8157945897251615
        ],
        [
            0.5109912902906809
        ],
        [
            0.23813856881441187
        ]
    ],
    "b2": [
        1.7132674824107537
    ]
}


class Player:
    def choose_action(self, board):
        raise NotImplementedError


class AIPlayer(Player):
    
    HIDDEN_NEURONS = 10

    def __init__(self, seed=None, weights=None):
        self.move_queue = []
        self.scorelist = []
        self.block_num = 0
        self.board_cells = None
        if weights: # for training
            self.w1 = weights['w1']
            self.b1 = weights['b1']
            self.w2 = weights['w2']
            self.b2 = weights['b2']
        else:
            self.w1 = BEST_WEIGHTS['w1']
            self.b1 = BEST_WEIGHTS['b1']
            self.w2 = BEST_WEIGHTS['w2']
            self.b2 = BEST_WEIGHTS['b2']

    def get_weights(self):
        return {'w1': self.w1, 'b1': self.b1, 'w2': self.w2, 'b2': self.b2}

    def print_board(self, board):
        print("--------")
        for y in range(24):
            s = ""
            for x in range(10):
                if (x,y) in board.cells:
                    s += "#"
                else:
                    s += "."
            print(s, y)
            
    def generate_moves(self, board):
        # For each possible move of the current piece, simulate all possible moves of the next piece
        moves_2d = []
        actions_2d = []
        for r1 in range(4):
            for x1 in range(10):
                first_board = board.clone()
                # first move
                for _ in range(r1):
                    first_board.falling.rotate(Rotation.Clockwise, first_board)
                dx1 = x1 - first_board.falling.left
                if dx1 < 0:
                    first_board.falling.move(Direction.Left, first_board, -dx1)
                elif dx1 > 0:
                    first_board.falling.move(Direction.Right, first_board, dx1)
                first_board.falling.move(Direction.Drop, first_board)
                first_board.land_block()

                #enumerate all possible next moves for each first move
                row_moves = []
                row_actions = []

                if first_board.falling is not None:
                    for r2 in range(4):
                        for x2 in range(10):
                            second_board = first_board.clone()
                            for _ in range(r2):
                                second_board.falling.rotate(Rotation.Clockwise, second_board)
                            dx2 = x2 - second_board.falling.left
                            if dx2 < 0:
                                second_board.falling.move(Direction.Left, second_board, -dx2)
                            elif dx2 > 0:
                                second_board.falling.move(Direction.Right, second_board, dx2)
                            second_board.falling.move(Direction.Drop, second_board)
                            second_board.land_block()
                            row_moves.append(second_board)
                            row_actions.append(((r1, x1), (r2, x2)))
                else:
                    # if no other block just append original
                    row_moves.append(first_board)
                    row_actions.append(((r1, x1), None))
                moves_2d.append(row_moves)
                actions_2d.append(row_actions)
        return moves_2d, actions_2d

    def get_features(self, move, board):

        holes = 0
        for x in range(10):
            found_block = False
            for y in range(24):
                if (x, y) in move.cells:
                    found_block = True
                elif found_block and (x, y) not in move.cells:
                    holes += 1

        heights = []
        for x in range(10):
            col_height = 0
            for y in range(24):
                if (x,y) in move.cells:
                    col_height = 24 - y
                    break
            heights.append(col_height)
        
        bump = 0
        for i in range(9):
            bump += abs(heights[i + 1] - heights[i])

        lines_cleared = (len(board.cells) + 4 - len(move.cells)) // move.width if move.width > 0 else 0

        # normalising features
        norm_holes = holes / 50.0
        norm_bump = bump / 100.0
        norm_max_height = max(heights) / 24.0
        height_var = max(heights) - min(heights)
        norm_height_var = height_var / 24.0
        norm_lines_cleared = lines_cleared / 4.0

        return (norm_holes, norm_bump, norm_max_height, norm_height_var, norm_lines_cleared, )
    
    def sgmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def fp(self, features):
        hidden_inputs = [0] * self.HIDDEN_NEURONS
        for i in range(self.HIDDEN_NEURONS):  
            for j in range(5): #each feature
                hidden_inputs[i] += features[j] * self.w1[j][i]
            hidden_inputs[i] += self.b1[i]

        # activation
        hidden_outputs = [self.sgmoid(x) for x in hidden_inputs]

        output = 0
        for i in range(self.HIDDEN_NEURONS):  
            output += hidden_outputs[i] * self.w2[i][0]
        output += self.b2[0]

        return output

    def get_score(self, move, board):
        features = self.get_features(move, board)
        # return -1*features[0]+ -1*features[1]+ -1*features[2]+ -1*features[3]+ 1*features[4]
        return self.fp(features)

    def score_moves(self, moves_2d, board):
        # moves_2d: 2D array of simulated boards
        scores_2d = []
        for row in moves_2d:
            row_scores = []
            for m in row:
                row_scores.append(self.get_score(m, board))
            scores_2d.append(row_scores)
        return scores_2d

    def choose_action(self, board):
        # self.print_board(board)
        if self.board_cells != board.cells:
            self.block_num += 1

            moves_2d, actions_2d = self.generate_moves(board)
            scores_2d = self.score_moves(moves_2d, board)
            best_score = None
            best_action = None
            for i, row in enumerate(scores_2d):
                for j, score in enumerate(row):
                    if (best_score is None) or (score > best_score):
                        best_score = score
                        best_action = actions_2d[i][j][0]  # use first move only for current piece
            self.scorelist.append(best_score)
            rotations, target_x = best_action
            for _ in range(rotations):
                self.move_queue.append(Rotation.Clockwise)
            temp_board = board.clone()
            for _ in range(rotations):
                temp_board.falling.rotate(Rotation.Clockwise, temp_board)
            dx = target_x - temp_board.falling.left
            if dx > 0:
                for _ in range(dx):
                    self.move_queue.append(Direction.Right)
            else:
                for _ in range(-dx):
                    self.move_queue.append(Direction.Left)
            self.board_cells = board.cells
        if not self.move_queue:
            return Direction.Down
        return self.move_queue.pop(0)


SelectedPlayer = AIPlayer