from board import Direction, Rotation, Action, Shape, Block

import math


BEST_WEIGHTS = {
    "w1": [
        [
            -0.1927062174432148,
            -1.8904515672692002,
            3.854917019406205,
            -0.3692542925261456,
            4.398636433050703,
            1.813256170879606
        ],
        [
            -1.6411909017881579,
            3.44766244961282,
            0.5565415547440347,
            0.1998394375500775,
            1.5108056015301843,
            1.9289656648563307
        ],
        [
            0.6865079594433822,
            0.4021603775944168,
            0.21249426728618814,
            5.209655190647702,
            3.3526950375578197,
            -0.6405203702747556
        ],
        [
            -0.18516251350800172,
            2.281472179486085,
            0.175179724309243,
            0.32120126193601173,
            -0.15305529321542732,
            0.49422206904756727
        ],
        [
            0.5637355778836036,
            -0.29696511521934055,
            -1.1254929629838517,
            1.2408765835490283,
            4.061394344655122,
            0.5468292648146064
        ],
        [
            -2.416529793326332,
            -0.2748876188592815,
            0.06856766303082051,
            2.600081762720423,
            0.9394715482340735,
            3.0045136175937857
        ],
        [
            0.02544398284597349,
            -0.26486298983913237,
            1.1344917125618994,
            0.4875675797223741,
            2.6681373472025838,
            -0.4000585631828416
        ]
    ],
    "b1": [
        -1.7102244699710405,
        2.1762622377706586,
        -1.533427543507537,
        0.3644381075506623,
        -0.3548203165863567,
        1.9419284599477047
    ],
    "w2": [
        [
            0.3727536554689621,
            -0.877485849533826,
            0.7550842741884309,
            2.725326944314095
        ],
        [
            -1.1059422715407012,
            0.10441837320248659,
            -3.015296536764048,
            0.8179927001067527
        ],
        [
            2.315801511790079,
            3.685800364208931,
            1.4244090418070328,
            -1.5581888666690002
        ],
        [
            -2.932015375126067,
            -0.9449908579589205,
            0.26509416172712197,
            -1.9759265464930238
        ],
        [
            -2.2697626042098977,
            1.166254347997198,
            1.5052533321023756,
            2.4663942284300004
        ],
        [
            -2.241610146305781,
            -0.4151450052478951,
            1.2892229939671185,
            6.049852833480946
        ]
    ],
    "b2": [
        0.3664655395280376,
        -2.256314264314429,
        0.6579677676498774,
        -2.029722775798487
    ],
    "w3": [
        [
            0.8521590985734687
        ],
        [
            -2.061004553639114
        ],
        [
            -1.633228095535831
        ],
        [
            -0.16820764979440328
        ]
    ],
    "b3": [
        1.1981144292187502
    ]
}


class Player:
    def choose_action(self, board):
        raise NotImplementedError


class AIPlayer(Player):
    HIDDEN_NEURONS_1 = 6
    HIDDEN_NEURONS_2 = 4

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
            self.w3 = weights['w3']
            self.b3 = weights['b3']
        else:
            self.w1 = BEST_WEIGHTS['w1']
            self.b1 = BEST_WEIGHTS['b1']
            self.w2 = BEST_WEIGHTS['w2']
            self.b2 = BEST_WEIGHTS['b2']
            self.w3 = BEST_WEIGHTS['w3']
            self.b3 = BEST_WEIGHTS['b3']

    def get_weights(self):
        return {'w1': self.w1, 'b1': self.b1, 'w2': self.w2, 'b2': self.b2, 'w3': self.w3, 'b3': self.b3}

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
        first_moves = []  # (sim_board, (r1, x1))
        for r1 in range(4):
            for x1 in range(10):
                first_board = board.clone()
                for _ in range(r1):
                    first_board.falling.rotate(Rotation.Clockwise, first_board)
                dx1 = x1 - first_board.falling.left
                if dx1 < 0:
                    first_board.falling.move(Direction.Left, first_board, -dx1)
                elif dx1 > 0:
                    first_board.falling.move(Direction.Right, first_board, dx1)
                first_board.falling.move(Direction.Drop, first_board)
                first_board.land_block()
                first_moves.append((first_board, (r1, x1)))

        # next block lookahead
        first_scores = [self.get_score(b, board) for b, _ in first_moves]
        if len(first_scores) > 3:
            top_indices = sorted(range(len(first_scores)), key=lambda i: first_scores[i], reverse=True)[:3]
        else:
            top_indices = list(range(len(first_scores)))

        moves_2d = []
        actions_2d = []
        for idx in top_indices:
            first_board, first_action = first_moves[idx]
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
                        row_actions.append((first_action, (r2, x2)))
            else:
                row_moves.append(first_board)
                row_actions.append((first_action, None))
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
        
        aggregate_height = sum(heights)

        # normalising features
        norm_holes = holes / 50.0
        norm_bump = bump / 100.0
        norm_max_height = max(heights) / 24.0
        height_var = max(heights) - min(heights)
        norm_height_var = height_var / 24.0
        norm_lines_cleared = lines_cleared / 4.0
        norm_blocks_placed = self.block_num / 400.0
        norm_aggregate_height = aggregate_height / 240.0  
        
        return (norm_holes, norm_bump, norm_max_height, norm_height_var, norm_lines_cleared, norm_blocks_placed, norm_aggregate_height)
    
    def sgmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def fp(self, features):
        # first hidden layer
        hidden1 = [0] * self.HIDDEN_NEURONS_1
        for i in range(self.HIDDEN_NEURONS_1):
            for j in range(7):  
                hidden1[i] += features[j] * self.w1[j][i]
            hidden1[i] += self.b1[i]
        hidden1 = [self.sgmoid(x) for x in hidden1]
        # second hidden layer
        hidden2 = [0] * self.HIDDEN_NEURONS_2
        for i in range(self.HIDDEN_NEURONS_2):
            for j in range(self.HIDDEN_NEURONS_1):
                hidden2[i] += hidden1[j] * self.w2[j][i]
            hidden2[i] += self.b2[i]
        hidden2 = [self.sgmoid(x) for x in hidden2]
        # Output
        output = 0
        for i in range(self.HIDDEN_NEURONS_2):
            output += hidden2[i] * self.w3[i][0]
        output += self.b3[0]
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
                        best_action = actions_2d[i][j][0]
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