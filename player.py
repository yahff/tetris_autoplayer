from board import Direction, Rotation, Action, Shape, Block

import math


BEST_WEIGHTS = {
    "w1": [
        [
            -2.573309983091046,
            -2.4462214426107725,
            -0.4301840064433958,
            2.5939096580686387,
            0.22721776749878253,
            0.7137412974852367
        ],
        [
            -2.08681576535045,
            -0.953323779500822,
            -0.8467576318864789,
            -1.8319126555432412,
            0.6069940039285604,
            0.8105435865635813
        ],
        [
            -0.396925491187047,
            -0.2623432300247547,
            -3.0294941718790733,
            -0.5229566583479186,
            -1.5066772232501684,
            -0.037806179532128725
        ],
        [
            0.14634967132423604,
            0.7728634317890821,
            1.4151645017121792,
            -0.3135935306964508,
            0.6236325349085157,
            -1.1324870662083382
        ],
        [
            1.948419780585665,
            -2.6554407554822372,
            -0.15878537203178067,
            -0.6438910367264266,
            -0.43416488312859114,
            2.013137238785161
        ],
        [
            0.9406311338559948,
            -0.8448510053257615,
            -1.2852665387317144,
            -1.8051666429121407,
            -3.5690984358916045,
            -0.39647106842947266
        ],
        [
            -0.9320919522417115,
            -2.120125156141931,
            -0.22365314506229583,
            0.816864568426312,
            -0.5267626502907774,
            1.5239318188766728
        ]
    ],
    "b1": [
        0.4984612404841975,
        0.37111628182251133,
        2.038649908735924,
        1.4529258913195737,
        0.02299439162899919,
        -2.479010688131089
    ],
    "w2": [
        [
            -1.3463216473915742,
            0.2944641902000372,
            -0.4315448648782705,
            0.0487326563663768
        ],
        [
            -1.4248739342460806,
            1.336420406108659,
            0.1865876584304993,
            0.345312487552232
        ],
        [
            -0.692999740520361,
            0.5673940905081433,
            -0.5821168141724635,
            0.7206231703441064
        ],
        [
            0.12133996296763472,
            -1.4835329051156676,
            0.35895976503878113,
            0.8451599976953714
        ],
        [
            0.6300977732753137,
            -0.29057110259484176,
            -0.5802974959564794,
            0.9554797993579186
        ],
        [
            0.14807896280180222,
            0.3596946771953812,
            0.6133087874387455,
            -0.6269785737414291
        ]
    ],
    "b2": [
        1.4779543081900068,
        0.35854284141954695,
        0.9028182907128297,
        -2.370918770388825
    ],
    "w3": [
        [
            -4.141751183915804
        ],
        [
            1.513111520253693
        ],
        [
            -0.3840884637791442
        ],
        [
            -0.29719113307848544
        ]
    ],
    "b3": [
        -0.2580355249851686
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
        self.will_die = False

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
        if len(first_scores) > 20:
            top_indices = sorted(range(len(first_scores)), key=lambda i: first_scores[i], reverse=True)[:20]
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
                        
                        # dummy piece to check collision
                        if second_board.next is None:
                            second_board.next = Block(Shape.I) 
                        
                        second_board.land_block()
                        
                        if not second_board.alive:
                            self.will_die = True

                        row_moves.append(second_board)
                        row_actions.append((first_action, (r2, x2)))
            else:
                row_moves.append(first_board)
                row_actions.append((first_action, None))
            moves_2d.append(row_moves)
            actions_2d.append(row_actions)
        return moves_2d, actions_2d

    def calculate_deepest_well(self, move):
        heights = []
        for x in range(10):
            col_height = 0
            for y in range(24):
                if (x, y) in move.cells:
                    col_height = 24 - y
                    break
            heights.append(col_height)
        max_well_depth = 0
        for i in range(10):
            if i == 0:
                left_height = 24
                right_height = heights[i + 1]
            elif i == 9:
                left_height = heights[i - 1]
                right_height = 24
            else:
                left_height = heights[i - 1]
                right_height = heights[i + 1]
            well_depth = min(left_height, right_height) - heights[i]
            max_well_depth = max(max_well_depth, well_depth)
        return max_well_depth

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
        well_depth = self.calculate_deepest_well(move)

        # normalising features
        norm_holes = holes / 50.0
        norm_bump = bump / 100.0
        norm_max_height = max(heights) / 24.0
        height_var = max(heights) - min(heights)
        norm_height_var = height_var / 24.0
        norm_lines_cleared = lines_cleared / 4.0
        norm_blocks_placed = self.block_num / 400.0
        norm_aggregate_height = aggregate_height / 240.0  
        norm_well_depth = well_depth / 24.0
        
        return (norm_holes, norm_bump, norm_max_height, norm_height_var, norm_lines_cleared, norm_blocks_placed, norm_aggregate_height, norm_well_depth)

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
        elif self.will_die:
            self.will_die = False
            self.move_queue.clear()
            return Action.Bomb
        return self.move_queue.pop(0)


SelectedPlayer = AIPlayer