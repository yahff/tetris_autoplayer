from board import Direction, Rotation, Action
import math


BEST_WEIGHTS = {
    "w1": [
        [
            -2.498394409679257,
            -0.2811040489761158,
            1.586258070340182,
            -0.1821008359939209,
            -0.5564545390871992,
            1.0592832531521106,
            -1.087434214846339,
            -1.9419908034776292,
            -0.8441329133811516,
            -1.543791179383101
        ],
        [
            0.6166180580462742,
            -0.1733354545452676,
            -1.544475176077628,
            0.8808929497591378,
            -0.9063588368318658,
            0.6735902038936761,
            -0.2859663508270044,
            -0.8704523369041814,
            -0.7956707024801686,
            -0.47840566631555337
        ],
        [
            1.1527352068560517,
            0.8661737918270644,
            0.45131928039776026,
            0.8794368827851127,
            -0.07648767560112854,
            1.0698956435205658,
            -0.14117026847264075,
            -1.6665109052861635,
            0.038896402621117876,
            -0.7388515037710335
        ],
        [
            1.2960880378133908,
            -0.25453083467033755,
            0.47639502415503954,
            -1.490478972616657,
            -0.6588495448086757,
            -1.1037844141724555,
            -0.6214223198818012,
            0.48805571108775725,
            -0.5999875202940618,
            -0.5044669118104351
        ],
        [
            -0.16828876896415773,
            -0.5313147921343582,
            -0.3911479567704982,
            -0.401227803890223,
            -2.597517298079481,
            -0.8501146195624624,
            1.1440937105988347,
            -0.20606474914314343,
            0.07673151142182077,
            0.20916063400489052
        ]
    ],
    "b1": [
        -0.3293400424636908,
        0.8722343849826841,
        -0.069045056659717,
        0.5276097002007806,
        -0.1846269548961998,
        0.26878402533251905,
        0.1024894462848637,
        -0.01813464587500299,
        -2.1022545750137485,
        -0.5439128414627694
    ],
    "w2": [
        [
            1.4566337942497611
        ],
        [
            -1.0661375541981233
        ],
        [
            -1.374993594195138
        ],
        [
            -0.816259602219227
        ],
        [
            2.1601312120275598
        ],
        [
            -1.428015480019449
        ],
        [
            1.0204130423174806
        ],
        [
            1.8239122406683954
        ],
        [
            -0.6109382355880408
        ],
        [
            0.7052487260023739
        ]
    ],
    "b2": [
        -0.8665448954748007
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