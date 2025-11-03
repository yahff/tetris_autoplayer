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
        move_list = []
        action_list = []
        for r in range(4):
            for target_x in range(10):
                test = board.clone()

                for _ in range(r):
                    test.falling.rotate(Rotation.Clockwise, test)
                
                current_x = test.falling.left

                if target_x < current_x:
                    test.falling.move(Direction.Left, test, current_x - target_x)
                elif target_x > current_x:
                    test.falling.move(Direction.Right, test, target_x - current_x)
                

                test.falling.move(Direction.Drop, test)
                test.land_block()
                



                action_list.append((r, target_x))
                move_list.append(test)

        return move_list, action_list
                
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

    def score_moves(self, moves, board):
        scores = []
        for m in moves:
            scores.append(self.get_score(m, board))
        return scores
        
    def choose_action(self, board):
        # self.print_board(board)
        if self.board_cells != board.cells:
            
            self.block_num += 1
            #print("block number " + str(self.block_num))

            possible_moves, actions = self.generate_moves(board)
            scores = self.score_moves(possible_moves, board)
            # print(max(scores))
            # print(min(scores))
            best = max(scores)
            self.scorelist.append(best)
            
            idx = scores.index(best)
            # self.print_board(possible_moves[idx])
            # print(min(scores))
            # print(max(scores))
            rotations = actions[idx][0]
            target_x = actions[idx][1]

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
                for _ in range(-1 * dx):
                    self.move_queue.append(Direction.Left)

            self.board_cells = board.cells


        if not self.move_queue:
            return Direction.Down
        return self.move_queue.pop(0)


SelectedPlayer = AIPlayer