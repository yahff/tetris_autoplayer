from board import Direction, Rotation, Action
import math


BEST_WEIGHTS = {
    "w1": [
        [
            -0.6338201526401543,
            0.3746836726324072,
            -1.8125958141702316,
            -1.1111465016428332,
            1.1537628161665956,
            1.6740468647728757,
            0.29332685543750703,
            1.396859785568565,
            0.7848925660913348,
            -1.4688031082315907
        ],
        [
            -0.736799655649035,
            -1.6535823201062598,
            1.3851636119810338,
            1.134325795714786,
            0.24999372559776356,
            0.9504842596027578,
            0.2552006206702844,
            -0.6599314516385135,
            1.5923176048529664,
            0.5021616980527543
        ],
        [
            1.6634321587445529,
            -0.8973341462537427,
            0.02641288714507306,
            -0.4433735404858292,
            0.8865872882970134,
            0.9964139070509952,
            -0.8216394835837884,
            -1.1317282905514667,
            -0.2531902457501154,
            -0.5290966568641902
        ],
        [
            0.25643851606432166,
            -1.7846989002151998,
            -1.81984715768994,
            -0.4656153481574825,
            -1.9596804616297452,
            -0.8410059159812886,
            0.04407585781439938,
            -1.3719696745106926,
            -1.983832748483287,
            -1.230309492036672
        ],
        [
            -0.15035709786277707,
            1.9018031766368404,
            1.274734025293267,
            -0.30181077337995754,
            -1.104534557877983,
            -1.693676411412448,
            -1.7896498104857792,
            0.6641979359552352,
            -0.6712529209722176,
            1.5773369577637717
        ],
        [
            1.119231994049568,
            1.8784180756186708,
            0.9003035867070994,
            -0.8659731257061236,
            -1.3226301857748566,
            1.159246220495283,
            0.01012017823578093,
            -1.4964439380023604,
            1.5904877152954024,
            -0.35903982049065686
        ],
        [
            1.5643656015371366,
            -1.3622606335381051,
            -0.7005845880068118,
            -0.29171879825138225,
            1.07088916463413,
            0.31395054573734305,
            0.48213864293900066,
            -1.7500902523264434,
            -1.958813952183314,
            -1.716309019790605
        ]
    ],
    "b1": [
        0.2856885982519244,
        -0.9604460417734759,
        0.7399174077841593,
        -0.05901612549151003,
        -0.9789196664580044,
        0.14605313921767205,
        -0.3809063815036051,
        0.3946948361598499,
        0.6128926924829792,
        -0.5430438065192722
    ],
    "w2": [
        [
            -1.583356225559203
        ],
        [
            0.5190294427909721
        ],
        [
            1.783325541217117
        ],
        [
            0.8359362833663271
        ],
        [
            -0.9387070686798928
        ],
        [
            -1.5666713245740986
        ],
        [
            1.4570203840322646
        ],
        [
            -0.9377657422730903
        ],
        [
            1.6177564898973844
        ],
        [
            0.3886020085583617
        ]
    ],
    "b2": [
        0.7134868944391792
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

        return (norm_holes, norm_bump, norm_max_height, norm_height_var, norm_lines_cleared)
    
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