from board import Direction, Rotation, Action
import math


BEST_WEIGHTS = {
    "w1": [
        [
            0.5394154362424486,
            -0.5505905418435032,
            -0.33804561223153384,
            -0.6973523966318007,
            0.1353887209894905,
            1.042483231205991,
            0.7150277685840276,
            -0.8606906840371078,
            0.03551666457797273,
            0.2528259025706425
        ],
        [
            -0.5419976588951221,
            -0.028989343165921597,
            -0.6502993098143772,
            0.6633318790811816,
            0.5626065813973754,
            0.9697602342930757,
            -0.5933273872051731,
            0.2446053349474427,
            -0.22296876782558842,
            0.6924961087641016
        ],
        [
            0.6756947235224123,
            1.1406118619440198,
            0.627608354612064,
            -0.4611964563970434,
            0.027155356674048822,
            -0.3134282738775121,
            0.3480520921534115,
            -0.327658796966075,
            -0.5672753279041388,
            -0.09669378831916017
        ],
        [
            0.2863373174335161,
            0.96021612218792,
            0.7320869617329592,
            -0.9459279705475716,
            -0.03619810319248795,
            -0.860280992076237,
            0.45159770170651903,
            0.1955661925061675,
            -0.18946733310688518,
            0.9430011464374951
        ],
        [
            0.1443566127940675,
            -0.2791358692840997,
            -0.5729881737435104,
            0.48003616446929587,
            -1.0608256115158958,
            -0.38912631879542714,
            0.1550081216055852,
            0.9978231815702816,
            0.35293835939281104,
            -0.3615135896397276
        ]
    ],
    "b1": [
        -0.45132998589423784,
        -0.25392843304177215,
        0.5912952927693093,
        -0.9180025523861972,
        -0.24815824665273917,
        -0.5100328997418616,
        -0.5516082228310785,
        -0.6744016644418485,
        -0.44559367782507875,
        -1.212678236347887
    ],
    "w2": [
        [
            -0.4712649765071261
        ],
        [
            0.2157630887546078
        ],
        [
            0.12081920186423924
        ],
        [
            0.3208996929088112
        ],
        [
            -0.729772509992027
        ],
        [
            -0.7937288387926272
        ],
        [
            -0.8258172399345683
        ],
        [
            0.17871188078892108
        ],
        [
            -0.026426150996157652
        ],
        [
            -0.32075718078815263
        ]
    ],
    "b2": [
        0.34772981889933274
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