from board import Direction, Rotation, Action
import math


BEST_WEIGHTS = {
    "w1": [
        [
            2.5618312189505548,
            0.6005209051123399,
            0.6997143224399228,
            2.594247793514352,
            1.1083844156719649,
            1.7789374273546985,
            1.1300257195071484,
            5.185728069328615,
            -0.6776293817722628,
            -0.1092546155682469
        ],
        [
            -0.1958368651660657,
            -0.36577885324394077,
            -0.514539545942354,
            -1.070066365693242,
            -0.09128880473507439,
            -1.5102300382215823,
            0.19689294055272122,
            -0.40729018001295403,
            -0.8256765999675673,
            -0.01613157601555948
        ],
        [
            -1.2106353897701014,
            1.655862168953595,
            -2.2653570921920405,
            -0.06767239155212257,
            -0.12839208089643062,
            -1.4455099879890128,
            2.3825335527866334,
            1.2843474368876822,
            -2.813450481389957,
            -1.72892915578735
        ],
        [
            2.497769904822287,
            -0.6626662465239719,
            1.3717457518860159,
            0.24397051700152006,
            -0.3197528234689635,
            -3.602672910867204,
            1.505864297623459,
            -2.297478793378604,
            2.5120276767045264,
            0.327167646692619
        ],
        [
            1.286758729523411,
            -1.495215669364692,
            0.9185277741781868,
            -1.6422161408936475,
            -0.7941467976590652,
            1.355422046302696,
            0.4305512714365807,
            -3.9112875710835144,
            0.34521630085560173,
            0.5449780879148437
        ]
    ],
    "b1": [
        -0.7607516362493253,
        1.1206539624290406,
        1.272363319492249,
        -1.4676869352560953,
        1.5355500809969413,
        -0.6059832853049532,
        -1.576477000191764,
        3.694516505991363,
        -0.2661182230515414,
        -0.14515168149034852
    ],
    "w2": [
        [
            -4.421140046841169
        ],
        [
            0.23220904036708578
        ],
        [
            2.865059948999287
        ],
        [
            -1.4833251649773218
        ],
        [
            0.9659977359070019
        ],
        [
            -0.5782716629857092
        ],
        [
            -0.11351452220241443
        ],
        [
            -3.033157328811676
        ],
        [
            2.3529191578279054
        ],
        [
            -1.9333139677249993
        ]
    ],
    "b2": [
        -0.5995137466263195
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