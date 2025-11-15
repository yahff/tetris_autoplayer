from board import Direction, Rotation, Action, Shape, Block

import math


LIN_WEIGHTS = {'w1': -50.066800807934655, 'w2': -6.212798702761036, 'w3': 25.29164492215014, 'w4': -2.4038140412102798, 'w5': 2.622652669017036, 'w6': 70.69099553242108, 'w7': -3.2772422719177845, 'w8': 1.3849998272188095, 'w9': -3.288347936236124, 'w10': -0.644814303528845, 'w11': 74.97737613503256, 'w12': 85.01843561952785, 'w13': 381.3570425650063, 'w14': 984.3407758574025, 'w15': -0.7130312412296305, 'w16': 5.140375053810649}


BEST_WEIGHTS = {
    "w1": [
        [
            0.29604549827790205,
            1.258728129872265,
            0.9912329987270857,
            0.08326189413025814,
            -0.14748758195549205,
            2.6795214087712234
        ],
        [
            -0.7573303530411369,
            0.7009172600358369,
            0.7818694904598618,
            -0.5145593967854523,
            2.1700864123321177,
            0.7337771628539529
        ],
        [
            0.7250389786465619,
            0.025333837581571983,
            -0.633497797326027,
            -0.6274595441279476,
            0.931019906116549,
            -0.4578401902660752
        ],
        [
            -0.1190074148577448,
            -1.0736366826538746,
            0.3942417148790909,
            -0.685431228236488,
            0.887377868427516,
            0.44416169725334015
        ],
        [
            1.4629637220477802,
            -0.17239203058545566,
            -0.21448132177442064,
            0.034207518884751875,
            -0.21148911047559887,
            -0.14804363271238435
        ],
        [
            0.04851670837195621,
            0.3919718917283363,
            0.06386842683053573,
            0.9895416233923112,
            -0.3193802650564215,
            0.8410899515825101
        ],
        [
            1.3115611253023625,
            0.4523049721470182,
            -0.5315056930577733,
            1.336384517716807,
            0.21665246940591526,
            1.242725944886951
        ]
    ],
    "b1": [
        0.3424491979102976,
        -0.17097930144815654,
        -0.9798269957141044,
        -0.7895673947595664,
        -2.8472604701034654,
        -1.773243322144733
    ],
    "w2": [
        [
            -0.5199888610080483,
            -0.5878569498469447,
            0.13774658636920117,
            1.1663295953556159
        ],
        [
            0.22067640441909486,
            1.1369607849903185,
            -0.06890142092200718,
            0.6104798986644884
        ],
        [
            0.19799056359636985,
            0.16272629378535985,
            0.11477007357578167,
            1.1321515719394903
        ],
        [
            -1.5196422085823391,
            1.602147515684234,
            0.39173175317053005,
            -0.5351616300342007
        ],
        [
            -0.7955500542381024,
            -0.3689354813200396,
            0.21834545659420118,
            0.8740074509719946
        ],
        [
            1.6355884398040206,
            0.6567301452697274,
            1.0781248225819555,
            0.4865571494942441
        ]
    ],
    "b2": [
        0.6597958380340501,
        1.0085649588003145,
        0.03996788633825221,
        0.40455204969355135
    ],
    "w3": [
        [
            -0.24220052150418836
        ],
        [
            -0.2715272260730398
        ],
        [
            -1.088448749141323
        ],
        [
            -2.5135438793998945
        ]
    ],
    "b3": [
        1.1564274588841619
    ]
}


class Player:
    def choose_action(self, board):
        raise NotImplementedError


class AIPlayer(Player):
    HIDDEN_NEURONS_1 = 6
    HIDDEN_NEURONS_2 = 4

    def __init__(self, seed=None, weights=None, lin_weights=None):
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
        
        if lin_weights:  # for linear weight training
            self.lin_weights = lin_weights
        else:
            self.lin_weights = LIN_WEIGHTS

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
        if len(first_scores) > 10:
            top_indices = sorted(range(len(first_scores)), key=lambda i: first_scores[i], reverse=True)[:10]
        else:
            top_indices = list(range(len(first_scores)))
        # top_indices = range(40)
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
                # left_height = 0
                # left_height = 0
            well_depth = min(left_height, right_height) - heights[i]
            max_well_depth = max(max_well_depth, well_depth)
        return max_well_depth

    def get_features(self, move, board):
        # Calculate column heights
        heights = []
        for x in range(10):
            col_height = 0
            for y in range(24):
                if (x, y) in move.cells:
                    col_height = 24 - y
                    break
            heights.append(col_height)
        
        # Count holes
        holes = 0
        for x in range(10):
            found_block = False
            for y in range(24):
                if (x, y) in move.cells:
                    found_block = True
                elif found_block and (x, y) not in move.cells:
                    holes += 1
        
        # Bumpiness (sum of absolute height differences)
        bumpiness = sum(abs(heights[i + 1] - heights[i]) for i in range(9))
        
        # Lines cleared calculation
        lines_cleared = max(0, (len(board.cells) + 4 - len(move.cells)) // 10)
        
        # Aggregate height
        aggregate_height = sum(heights)
        max_height = max(heights)
        
        # Potential clears (count rows that are almost complete)
        potential_clears = 0
        for y in range(24):
            filled = sum(1 for x in range(10) if (x, y) in move.cells)
            if filled >= 8:  # Row is at least 80% full
                potential_clears += 1
        
        return {
            'holes': holes,
            'bumpiness': bumpiness,
            'max_height': max_height,
            'aggregate_height': aggregate_height,
            'lines_cleared': lines_cleared,
            'potential_clears': potential_clears
        }

    def sgmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def fp(self, features):
        # first hidden layer
        hidden1 = [0] * self.HIDDEN_NEURONS_1
        for i in range(self.HIDDEN_NEURONS_1):
            for j in range(8): 
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
        w = self.lin_weights
        score = 0
        
        # W1 x holes
        score += w["w1"] * features['holes']
        
        # W2 x bumpiness
        score += w["w2"] * features['bumpiness']
        
        # If max_height > w3: w4 * (max_height - w3) ^ w5
        if features['max_height'] > w["w3"]:
            score += w["w4"] * (features['max_height'] - w["w3"]) ** w["w5"]
        
        # If aggregate_height > w6: w7 * (aggregate_height - w6) ^ w8
        if features['aggregate_height'] > w["w6"]:
            score += w["w7"] * (features['aggregate_height'] - w["w6"]) ** w["w8"]
        
        # If potential_clears < 4: w9 * potential_clears ^ w10 (safe for 0^negative)
        if features['potential_clears'] < 4:
            base = features['potential_clears']
            exp = w["w10"]
            if base == 0 and exp < 0:
                safe_pow = 0 
            else:
                safe_pow = base ** exp
            score += w["w9"] * safe_pow
        
        # Line clear rewards
        lines = features['lines_cleared']
        if lines == 1:
            score += w["w11"]
        elif lines == 2:
            score += w["w12"]
        elif lines == 3:
            score += w["w13"]
        elif lines >= 4:
            score += w["w14"]
        
        # W15 * max_height (promote growing before reaching threshold w3)
        score += w["w15"] * features['max_height']
        
        # W16 * aggregate_height
        score += w["w16"] * features['aggregate_height']
        
        return score

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