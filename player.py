from board import Direction, Rotation, Action
import math


BEST_WEIGHTS = {
    "w1": [
        [
            0.5048737949108735,
            -5.492231312669914,
            1.9981010335074267,
            -1.5397401600976304,
            2.393386393126837,
            -1.5263707005039682,
            -4.90238784624775,
            -2.0823149635331903,
            -3.1360275950425605,
            -0.3848628124126694
        ],
        [
            0.5663994249816403,
            -0.7881256581808965,
            -0.7255770839564091,
            -2.103979831001074,
            -1.5468106139227886,
            -2.592220740096196,
            2.357893325018439,
            -0.9139154563723646,
            0.5492856627875458,
            1.7654147468281547
        ],
        [
            -1.07564702002183,
            -3.1196396361932655,
            -1.2140008216313007,
            2.839050032969105,
            -1.5376895926285434,
            1.8815361803505288,
            0.05762272326340201,
            -3.1515472814597705,
            1.491041547700363,
            4.590045341172667
        ],
        [
            -0.9752188654936201,
            2.6371782052701933,
            -1.0634432622702166,
            -1.864149435455316,
            3.104725971325624,
            -2.723010842435383,
            2.682662682662423,
            1.9047373770809661,
            -0.6068599242104277,
            3.16428303952405
        ],
        [
            -0.9623807185643628,
            0.9004886051016896,
            -2.5647703108760163,
            -1.2392332065203342,
            0.2084711822530403,
            1.5895938708152961,
            2.2833730686642566,
            -0.22191652360678382,
            0.49572300426330684,
            -2.200234037819817
        ]
    ],
    "b1": [
        0.8229943804698495,
        0.23861383261907498,
        -1.0962728905774415,
        3.104906137203656,
        -1.6892663948166255,
        -1.3516627123041456,
        5.831747642999843,
        1.1205818591964118,
        -0.35384675927693054,
        1.545368109208701
    ],
    "w2": [
        [
            -0.1485363810503656
        ],
        [
            4.039977675612165
        ],
        [
            1.02192126715817
        ],
        [
            -0.4889624593568097
        ],
        [
            1.576204023301726
        ],
        [
            -1.392375161380503
        ],
        [
            0.552857687754932
        ],
        [
            4.902722091525371
        ],
        [
            0.7702616134084319
        ],
        [
            2.7005219633836393
        ]
    ],
    "b2": [
        4.649329424288088
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

        first_scores = [self.get_score(b, board) for b, _ in first_moves]
        # get indices of top 5 moves only
        if len(first_scores) > 5:
            top_indices = sorted(range(len(first_scores)), key=lambda i: first_scores[i], reverse=True)[:5]
        else:
            top_indices = list(range(len(first_scores)))

        moves_2d = []
        actions_2d = []
        for idx in top_indices:
            first_board, (r1, x1) = first_moves[idx]
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