import copy


@staticmethod
def NumberOfAgents(board):
    agents = 0
    for i in range(board.rows):
        for j in range(board.cols):
            if board[i][j] == 1:
                agents += 1
    return agents


class Board:
    def _init_(self, board):
        self.cols = len(board)
        self.rows = len(board[0])
        self.board = board
        self.numberOfAgents = self.NumberOfAgents()
        self.cost = 0
        self.heuristics = 0
        self.agents_arrived = []

    def calc_arrived_agents(self, goal_board):
        for i in range(self.rows):
            for j in range(self.cols):
                if self.board[i][j] == 2 and goal_board.board [i][j] == 2:
                    self.agents_arrived.append((i, j))


    def get_heuristics(self):
        return self.heuristics + self.cost

    def validMoveRight(self, i, j):
        return (j != self.cols - 1) and (self.board[i][j + 1] == 0)

    def validMoveUp(self, i, j):
        return (i != 0) and (self.board[i - 1][j] == 0)

    def validMoveLeft(self, i, j):
        return (j != 0) and (self.board[i][j - 1] == 0)

    def validMoveDown(self, i, j):
        return (i == self.rows - 1) or (self.board[i + 1][j] == 0)

    def NumberOfAgents(self):
        agents = 0
        for i in range(self.rows):
            for j in range(self.cols):
                if self.board[i][j] == 2:
                    agents += 1
        return agents

    def _eq_(self, other):
        for i in range(self.rows):
            for j in range(self.cols):
                if self.board[i][j] != other.board[i][j]:
                    return False
        return True

    # def checkAdjecentAgents(self, other, i, j):
    #     if (other.board[i + 1][j] == 1) or (other.board[i - 1][j] == 1) or (other.board[i][j + 1] == 1) or (
    #             other.board[i][j - 1] == 1):
    #         return 1
    #     else:
    #         return 0

    def calc_heuristics(self, goal_board):
        # my heuristics takes the board, removes the blocks (@),
        # finds the nearest agent on the goal board not in use and matches between them.
        # then sums up the paths and returns a number, the smaller the mumber the better the board.
        # this heuristics is admissible (const???)
        for i in range(self.rows):
            for j in range(self.cols):
                if self.board[i][j] == 2:
                    dist = 1000
                    chosenK = None
                    k_used = []
                    for k in goal_board.agents:
                        if abs(k[0] - i) + abs(k[1] - j) < dist and k not in k_used and k not in goal_board.usedAgents:
                            dist = abs(k[0] - i) + abs(k[1] - j)
                            chosenK = k
                    k_used.append(chosenK)
                    self.heuristics += dist


class Goal_board(Board):
    def _init_(self, board):
        super()._init_(board)
        self.agents = self.find_agents()
        self.usedAgents = []

    def find_agents(self):
        agents = []
        for i in range(self.rows):
            for j in range(self.cols):
                if self.board[i][j] == 2:
                    agents.append((i, j))
        return agents

    def find_used_agents(self, current_board):
        for i in range(self.rows):
            for j in range(self.cols):
                if self.board[i][j] == 2 and current_board.board [i][j] == 2:
                    self.usedAgents.append((i, j))



class Game:
    def _init_(self, start_board, goal_board):
        self.all_available_moves = []
        self.errors = []
        self.adjacentAgents = []
        self.boards_visited = [start_board]
        self.stat_board = start_board
        self.goal_board = goal_board
        self.current_board = start_board
        self.validGame = self.checkValidGame()
        self.finised = self.checkIfFinished()
        self.iteration = 0
        self.play()

    def checkIfFinished(self):
        return self.current_board == self.goal_board

    def checkValidGame(self):
        return self.current_board.numberOfAgents >= self.goal_board.numberOfAgents

    def moveRight(self, i, j, iteration):
        tmp_borad = copy.deepcopy(self.current_board)
        tmp_borad.board[i][j] = 0
        tmp_borad.board[i][j + 1] = 2
        tmp_borad.cost = iteration
        tmp_borad.calc_heuristics(self.goal_board)
        return tmp_borad

    def moveUp(self, i, j, iteration):
        tmp_borad = copy.deepcopy(self.current_board)
        tmp_borad.board[i][j] = 0
        tmp_borad.board[i - 1][j] = 2
        tmp_borad.cost = iteration
        tmp_borad.calc_heuristics(self.goal_board)
        return tmp_borad

    def moveLeft(self, i, j, iteration):
        tmp_borad = copy.deepcopy(self.current_board)
        tmp_borad.board[i][j] = 0
        tmp_borad.board[i][j - 1] = 2
        tmp_borad.cost = iteration
        tmp_borad.calc_heuristics(self.goal_board)
        return tmp_borad

    def moveDown(self, i, j, iteration):
        tmp_borad = copy.deepcopy(self.current_board)
        if i == 5:
            tmp_borad.board[i][j] = 0
        else:
            tmp_borad.board[i][j] = 0
            tmp_borad.board[i + 1][j] = 2
        tmp_borad.cost = iteration
        tmp_borad.calc_heuristics(self.goal_board)
        return tmp_borad

    # def checkMoves(self):
    #     for i in self.current_board.rows:
    #         for j in self.current_board.cols:
    #             if self.current_board[i][j] == 1:
    #                 if self.current_board.validMoveRight(self.current_board, i, j):
    #                     self.all_available_moves.append(self.moveRight(self.current_board, i, j))

    def findMoves(self):
        for i in range(self.current_board.rows):
            for j in range(self.current_board.cols):
                if self.current_board.board[i][j] == 2 and self.goal_board.board[i][j] != 2:
                    if self.current_board.validMoveRight(i, j):
                        self.all_available_moves.append(self.moveRight(i, j, self.iteration))
                    if self.current_board.validMoveUp(i, j):
                        self.all_available_moves.append(self.moveUp(i, j, self.iteration))
                    if self.current_board.validMoveLeft(i, j):
                        self.all_available_moves.append(self.moveLeft(i, j, self.iteration))
                    if self.current_board.validMoveDown(i, j):
                        self.all_available_moves.append(self.moveDown(i, j, self.iteration))

    def sortMOves(self):
        self.all_available_moves.sort(key=lambda x: x.cost + x.heuristics)

    # def findAdjacentAgents(self):
    #     for x in self.all_available_moves:
    #         adjacent_agents_on_x = 0
    #         for i in range(x.rows):
    #             for j in range(x.cols):
    #                 if x.board[i][j] == 1:
    #                     adjacent_agents_on_x = adjacent_agents_on_x + x.checkAdjecentAgents(self.goal_board, i, j)
    #         self.adjacentAgents.append(adjacent_agents_on_x)
    #
    # def findSumOfErrors(self):
    #     for x in self.all_available_moves:
    #         errorsOnX = 0
    #         for i in range(x.rows):
    #             for j in range(x.cols):
    #                 if x.board[i][j] == 1 and self.goal_board.board[i][j] != 1:
    #                     errorsOnX += 1
    #         self.errors.append(errorsOnX)

    def move(self):
        visited = False
        bad_move = False
        self.all_available_moves[0].calc_arrived_agents(self.goal_board)
        self.current_board.calc_arrived_agents(self.goal_board)
        for i in self.current_board.agents_arrived:
            if i not in self.all_available_moves[0].agents_arrived:
                bad_move = True
        for k in self.boards_visited:
            if self.all_available_moves[0] == k:
                visited = True
                break
        if not visited and not bad_move:
            self.current_board = self.all_available_moves.pop(0)
            self.boards_visited.append(self.current_board)
            self.goal_board.find_used_agents(self.current_board)
            self.current_board.heuristics = 0
            self.current_board.cost = 0
            self.finised = self.checkIfFinished()
            print(self.current_board.board)
        else:
            self.iteration = -1
            self.all_available_moves.pop(0)


    def play(self):
        while (self.validGame) and (self.finised is False):
            self.iteration += 1
            self.findMoves()
            self.sortMOves()
            self.move()
            self.checkValidGame()
        print('finished')


if _name_ == "_main_":
    print('Hi Idan')
    board1 = Board(
        [[2, 0, 2, 0, 0, 2],
         [0, 0, 0, 2, 1, 0],
         [1, 0, 0, 0, 0, 2],
         [0, 0, 1, 0, 1, 0],
         [0, 0, 0, 0, 0, 0],
         [2, 1, 0, 0, 0, 0]])
    board2 = Goal_board(
        [[2, 0, 2, 0, 0, 0],
         [0, 0, 0, 2, 1, 2],
         [1, 0, 0, 0, 0, 2],
         [0, 0, 1, 0, 1, 0],
         [0, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0]])
    my_game = Game(board1, board2)
    # print (my_game.errorsOnCurrent)
    # print (my_game.errors)