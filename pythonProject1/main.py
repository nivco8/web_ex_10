import copy

#--------------------  print functions  ----------------------

#set a list of boards to print (the correct path)
def restore_solution(node):
    result = []
    while node.father != None:
        result.insert(0, node)
        node = node.father
    result.insert(0, node)
    return result

#print solution
def print_solution(solution, detail_output):
    if detail_output:
        for i in range(len(solution)):
            board = solution[i]
            if i == 0:
                print("Board 1 (starting position):")
                print()
                print_current_board(board.myBoard)
                print("-----------")
                print()
            elif i == len(solution) - 1:
                print('Board ' + str(i + 1) + " (goal position):")
                print_current_board(board.myBoard)
                print()
                print('Heuristic: ' + str(board.h_n))
            else:
                print('Board ' +str (i+1) + ":")
                print_current_board(board.myBoard)
                print()
                print('Heuristic: ' + str(board.h_n))
                print("-----------")
                print()
    else:
        for i in range(len(solution)):
            board = solution[i]
            if i == 0:
                print("Board 1 (starting position):")
                print_current_board(board.myBoard)
                print("-----------")
                print()
            elif i == len(solution) - 1:
                print('Board ' + str(i + 1) + " (goal position):")
                print_current_board(board.myBoard)
            else:
                print('Board ' + str(i + 1) + ":")
                print_current_board(board.myBoard)
                print("-----------")
                print()


# assistance function to print board
def print_current_board(board):
    print(" ", end="  ")
    for i in range(0, 6):
        print(i + 1, end="  ")
    print("")
    for i in range(0, 6):
        print(i + 1, end=" ")
        for j in range(0, 6):
            if board[i][j] == 1:
                print(" @", end=" ")
            elif board[i][j] == 2:
                print(" *", end=" ")
            else:
                print("  ", end=" ")
        print("")



#------------------------   Class Board    --------------------------------------

class Board:
    # board constructor
    def __init__(self, board, cost):
        self.father = None
        self.myBoard = board
        self.numOfCol = len(board)
        self.numOfRow = len(board[0])
        self.numOfAgents = self.check_num_of_agents()
        self.g_n = cost
        self.h_n = 0
        self.f_n = 0
        self.neighbors = []


    #check if moves are possible
    def check_right(self, i, j):
        if j == self.numOfCol-1 or self.myBoard[i][j+1] != 0:
            return False
        return True

    def check_left(self, i, j):
        if j == 0 or self.myBoard[i][j-1] != 0:
            return False
        return True

    def check_up(self, i, j):
        if i == 0 or self.myBoard[i-1][j] != 0:
            return False
        return True

    def check_down(self, i, j):
        if i != self.numOfRow - 1 and self.myBoard[i+1][j] != 0:
            return False
        return True


    #calc variables
    def calcToGoal(self, goal_board):
        self.h_n = self.set_h_n(goal_board.myBoard)
        self.set_total()
        return


    #set h_n
    def set_h_n(self, goal_board): #h(n)
        total_cost = 0
        for i in range(self.numOfRow):
            for j in range(self.numOfCol):
                if goal_board[i][j] == 2:
                    cost = self.closest_agent_cost(i, j)
                    total_cost += cost
        return total_cost


    #check which agent is the closest to (x,y)
    def closest_agent_cost(self, x, y):
        cost = 100
        countAgent = 0
        for i in range(self.numOfRow):
            for j in range(self.numOfCol):
                if self.myBoard[i][j] == 2:
                    countAgent += 1
                    if abs(x - i) + abs(y - j) < cost:
                        cost = abs(x - i) + abs(y - j)
        if countAgent == 0:
            return 0
        return cost


    #set possible neighboring boards
    def find_neighbors(self):
        for i in range(self.numOfRow):
            for j in range(self.numOfCol):
                if self.myBoard[i][j] == 2:
                    if self.check_right(i, j):
                        self.set_new_neighbor(i, j, 1)
                    if self.check_left(i, j):
                        self.set_new_neighbor(i, j, 2)
                    if self.check_up(i, j):
                        self.set_new_neighbor(i, j, 3)
                    if self.check_down(i, j):
                        self.set_new_neighbor(i, j, 4)

    #internal function of find neighbors
    def set_new_neighbor(self, i, j, source):
        temp_board = Board(copy.deepcopy(self.myBoard), self.g_n + 1) #neighbor board should get cost + 1
        #set neighbor for each situation (right, left, up, down)
        if source == 1:
            temp_board.myBoard[i][j] = 0
            temp_board.myBoard[i][j+1] = 2
            self.neighbors.append(temp_board)
        if source == 2:
            temp_board.myBoard[i][j] = 0
            temp_board.myBoard[i][j-1] = 2
            self.neighbors.append(temp_board)
        if source == 3:
            temp_board.myBoard[i][j] = 0
            temp_board.myBoard[i-1][j] = 2
            self.neighbors.append(temp_board)
        if source == 4:
            if i == self.numOfRow - 1:
                temp_board.myBoard[i][j] = 0
            else:
                temp_board.myBoard[i][j] = 0
                temp_board.myBoard[i+1][j] = 2
            self.neighbors.append(temp_board)

    #count number of agents on board
    def check_num_of_agents(self):
        agentCounter = 0
        for i in range(self.numOfRow):
            for j in range(self.numOfCol):
                if self.myBoard[i][j] == 2:
                    agentCounter += 1
        return agentCounter

    #calculate f_n
    def set_total(self):
        self.f_n = self.h_n + self.g_n
        return

    #set father for node
    def add_node_as_father(self, node):
        self.father = node


    def __hash__(self):
        return hash((self.myBoard))


    def __eq__(self, other):
        if other is None:
            return False
        for i in range(self.numOfRow):
            for j in range(self.numOfCol):
                if self.myBoard[i][j] != other.myBoard[i][j]:
                    return False
        return True


#-------------------------------------------------------------------


#check better node in frontier
def replace_node(frontier, child):
    for i in range(len(frontier)):
        if frontier[i] == child:
            frontier.pop(i)
            frontier.insert(i, child)
    return frontier


#check better g_n between boards
def isHigher(frontier, child):
    for i in range(len(frontier)):
        if child.myBoard == frontier[i].myBoard:
            if child.g_n < frontier[i].g_n:
                return True
    return False


#-----------------      A* ALGHORITHM      ----------------------

def Astars(starting_board, goal_board, detail_output):
    node = starting_board
    frontier = []
    explored = []
    frontier.append(node)
    while len(frontier) > 0:
        frontier.sort(key=lambda x: x.f_n) #make the first board in the frontier to be the one with the lowest f_n
        node = frontier.pop(0)
        if node == goal_board:
            solution = restore_solution(node)
            print_solution(solution, detail_output)
            return True
        node.find_neighbors() #set neighbors to the node
        explored.append(node)
        for child in node.neighbors:
            child.calcToGoal(goal_board) #calculate heuristic to each child
            child.add_node_as_father(node) #set current node as father of each child
            if child not in explored and child not in frontier:
                frontier.append(child)
            elif child in frontier:
                if isHigher(frontier, child):
                    frontier = replace_node(frontier, child)
    print("No path found.")


#check if board is valid
def check_valid(starting_board, goal_board):
    if starting_board.numOfAgents < goal_board.numOfAgents:
        return False
    if starting_board.numOfRow != 6 or starting_board.numOfCol != 6:
        return False
    if goal_board.numOfRow != 6 or goal_board.numOfCol != 6:
        return False
    for i in range(starting_board.numOfRow):
        for j in range(starting_board.numOfCol):
            if (starting_board.myBoard[i][j] == 1 and goal_board.myBoard[i][j] != 1):
                return False
            if (goal_board.myBoard[i][j] == 1 and starting_board.myBoard[i][j] != 1):
                return False
    return True


def find_path(starting_board, goal_board, search_method, detail_output):
    if search_method == 1:
        if check_valid(starting_board, goal_board):
            solve = Astars(starting_board, goal_board, detail_output)
        else:
            print ("Warning: unvalid boards")


if __name__ == "__main__":

#--------boards in the exercise:

    starting_board = (
        [[2, 0, 2, 0, 2, 0],
         [0, 0, 0, 2, 1, 2],
         [1, 0, 0, 0, 0, 0],
         [0, 0, 1, 0, 1, 0],
         [2, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0]])
    goal_board = (
        [[2, 0, 2, 0, 0, 0],
         [0, 0, 0, 2, 1, 2],
         [1, 0, 0, 0, 0, 2],
         [0, 0, 1, 0, 1, 0],
         [0, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0]])

    starting_board = Board(starting_board, 0)
    goal_board = Board(goal_board, 0)

    find_path(starting_board, goal_board, 1, True)