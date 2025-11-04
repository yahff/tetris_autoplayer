from board import Direction, Rotation, Action, Shape, Block

import math


BEST_WEIGHTS = {
    "w1": [
        [
            -1.1835961451451742,
            1.202704810893144,
            -1.6356226519354693,
            -5.930551071985017,
            -0.4355109318595083,
            3.819259626404543,
            1.636868656155593,
            -3.591994174357847,
            0.6700058757050474,
            0.34452461329875905
        ],
        [
            -4.352970054092909,
            3.3005937589441494,
            0.5820711701508127,
            1.5554272327283716,
            -0.35566991950057747,
            -0.7977752187289501,
            -2.2949964740317252,
            -2.577126860390476,
            4.682540535586572,
            3.0455671894428087
        ],
        [
            2.1261471529021905,
            -1.5461159550752952,
            -0.6844859964948461,
            -2.0919528047848024,
            0.1575038076698171,
            -1.081949983800732,
            0.9060370382770703,
            0.13915700404127285,
            3.4063452612600846,
            -0.5987811958760434
        ],
        [
            2.396569811897969,
            2.4443339441327208,
            2.0990550821817466,
            0.19028597367624023,
            -0.5029225082943397,
            -3.7227849905915624,
            1.442422720947865,
            -2.506298602788796,
            -1.4134721257127416,
            -2.941546598729561
        ],
        [
            0.8271602756477602,
            -2.8288466818556657,
            -0.8136612955393888,
            -6.750884307856305,
            -1.8672543262741117,
            0.23061840106965104,
            -1.6911805693411845,
            2.7761497143388056,
            -0.41487562028568115,
            -1.0685725219866824
        ],
        [
            -0.3483971744570251,
            -3.191325965468982,
            -2.272904694187633,
            -0.8888183793198599,
            1.8006266784957472,
            -2.4026369157659415,
            0.35709067516667714,
            -1.3834956805313183,
            1.3985913465241482,
            1.4075079023669523
        ]
    ],
    "b1": [
        -1.8743572713759347,
        -0.5712287690383885,
        -1.7016494616689437,
        0.359735674750869,
        -2.724524436477339,
        -0.37561421600188094,
        3.091303271447589,
        -2.1778156021849773,
        -1.4715657555006179,
        -0.5877844887167009
    ],
    "w2": [
        [
            -0.3845185719535079
        ],
        [
            -0.6119317821459572
        ],
        [
            -0.39332601266362477
        ],
        [
            2.2441040027704884
        ],
        [
            0.026903704194055966
        ],
        [
            -2.3371492081888725
        ],
        [
            -0.6645523301291176
        ],
        [
            1.8552026225275235
        ],
        [
            -1.0894794955257425
        ],
        [
            -0.8364735100732846
        ]
    ],
    "b2": [
        -3.900883693626965
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

        # add discard as a move if available
        if board.discards_remaining > 0:
            discard_board = board.clone()
            discard_board.discards_remaining -= 1
            discard_board.place_next_block()  # disc current, place next
            first_moves.append((discard_board, 'DISCARD'))

        # add bomb as move if available
        if board.bombs_remaining > 0 and board.next is not None and board.next.shape != 'B':
            bomb_board = board.clone()
            bomb_board.bombs_remaining -= 1
            # Set the next piece to a bomb
            bomb_board.next = Block(Shape.B)
            first_moves.append((bomb_board, 'BOMB'))

        #return if next block lookahead is not being used
        moves_2d = []
        actions_2d = []
        for first_board, first_action in first_moves:
            moves_2d.append([first_board])
            actions_2d.append([(first_action, None)])
        return moves_2d, actions_2d

        # next block lookahead
        first_scores = [self.get_score(b, board) for b, _ in first_moves]
        if len(first_scores) > 5:
            top_indices = sorted(range(len(first_scores)), key=lambda i: first_scores[i], reverse=True)[:4]
        else:
            top_indices = list(range(len(first_scores)))

        moves_2d = []
        actions_2d = []
        for idx in top_indices:
            first_board, first_action = first_moves[idx]
            row_moves = []
            row_actions = []
            if first_action == 'DISCARD':
                row_moves.append(first_board)
                row_actions.append(('DISCARD', None))
            elif first_action == 'BOMB':
                #all possible placements for the bomb piece
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
                            row_actions.append(('BOMB', (r2, x2)))
                else:
                    row_moves.append(first_board)
                    row_actions.append(('BOMB', None))
            else:
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

        # normalising features
        norm_holes = holes / 50.0
        norm_bump = bump / 100.0
        norm_max_height = max(heights) / 24.0
        height_var = max(heights) - min(heights)
        norm_height_var = height_var / 24.0
        norm_lines_cleared = lines_cleared / 4.0
        norm_blocks_placed = self.block_num / 400.0

        return (norm_holes, norm_bump, norm_max_height, norm_height_var, norm_lines_cleared, norm_blocks_placed)
    
    def sgmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def fp(self, features):
        hidden_inputs = [0] * self.HIDDEN_NEURONS
        for i in range(self.HIDDEN_NEURONS):  
            for j in range(6): #each feature
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
                    next = actions_2d[i][j][0]
                    if next == 'BOMB':
                        score /= 0.9 #decrease frequency of bomb uses
                    if next == 'DISCARD':
                        score /= 0.99 #decrease frequency of discard uses
                    if (best_score is None) or (score > best_score):
                        best_score = score
                        best_action = next
            self.scorelist.append(best_score)
            if best_action == 'DISCARD':
                return Action.Discard
            if best_action == 'BOMB':
                return Action.Bomb
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