import copy
import random
import math

#-------------------------------------------  Algorithms  ----------------------------------------------


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
            print_solution(solution, detail_output, 1)
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


#-----------------      HILL CLIMBING ALGHORITHM      ----------------------


def hill_climbing(starting_board, goal_board, detail_output):
    explored = []
    for i in range(5):
        current = copy.deepcopy(starting_board)
        current.calcToGoal(goal_board)
        finished_iter = False
        while not finished_iter:
            if current == goal_board:
                solution = restore_solution(current)
                print_solution(solution, detail_output, 1)
                return True
            current.find_neighbors()
            for neighbor in current.neighbors:
                neighbor.calcToGoal(goal_board)  # calculate heuristic to each child
                neighbor.add_node_as_father(current)  # set current node as father of each child
            current.neighbors.sort(key=lambda x: x.h_n)  # make the first board in the frontier to be the one with the lowest f_n
            for neighbor in current.neighbors:
                if neighbor not in explored:
                        if neighbor.h_n <= current.h_n:
                            explored.append(neighbor)
                            current = neighbor
                            break
                        else:
                            explored.append(neighbor)
                            finished_iter = True
                            break
                else:
                    if neighbor == current.neighbors[len(current.neighbors)-1]:
                        finished_iter = True
                        break
                    continue
    print("No path found.")


#-----------------      SIMULATED-ANNEALING      ----------------------

#schedule the T value of the simulated anealing algo
def schedule(iteration):
    if iteration == 100:
        return 0
    if iteration < 20:
        return 1
    elif iteration >= 20 and iteration < 30:
        return 0.9
    elif iteration >= 30 and iteration < 40:
        return 0.8
    elif iteration >= 40 and iteration < 50:
        return 0.7
    elif iteration >= 50 and iteration < 60:
        return 0.6
    else:
        return 0.4



def simulated_annealing(starting_board, goal_board, detail_output):
    current = starting_board
    for t in range(0, 101):
        if current == goal_board:
            solution = restore_solution(current)
            print_solution(solution, detail_output, 2)
            return True
        T = schedule(t) #set the T value
        if T == 0: #iteration 100
            print("No path found.")
            return False
        current.find_neighbors()
        for neighbor in current.neighbors:
            neighbor.calcToGoal(goal_board)  # calculate heuristic to each child
            neighbor.add_node_as_father(current)  # set current node as father of each child
        next_node = current.neighbors[random.randrange(0, len(current.neighbors))]
        current.considered_actions.append(next_node.restore_move)
        delta_e = current.h_n - next_node.h_n #the difference between current's heuristic to the next node's heuristics
        if delta_e > 0: #the step improve our h(n)
            prob_T = 1
            current.probabilities.append(prob_T)
            current = next_node
        else:
            random_float = random.random()
            prob_T = math.exp(delta_e / T)
            current.probabilities.append(prob_T)
            if random_float <= prob_T: #a little probability to take a bad step
                current = next_node


#-----------------      K-BEAM ALGHORITHM      ----------------------


def k_beam(k, starting_board, goal_board, detail_output):
    starting_board = starting_board
    explored = []
    all_neighbors = []
    finished = False
    first_iter = True
    while not finished:
        if first_iter:
            first_iter = False
            if starting_board == goal_board:
                solution = restore_solution(starting_board)
                print_solution(solution, detail_output)
                return True
            starting_board.find_neighbors()
            for neighbor in starting_board.neighbors:
                neighbor.calcToGoal(goal_board)  # calculate heuristic to each child
                neighbor.add_node_as_father(starting_board)  # set current node as father of each child
                all_neighbors.append(neighbor)
            explored.append(starting_board)
        all_neighbors.sort(key=lambda x: x.h_n) #sort all neighbors by h(n)
        k_neighbors = []
        for neighbor in all_neighbors:
            if len(k_neighbors) < k and neighbor not in explored: #if we have less than k neighbors that still not in explored list
                k_neighbors.append(neighbor)
                explored.append(neighbor)
        all_neighbors = []
        if len(k_neighbors) == 0:
            finished = True
        else:
            for beam in k_neighbors:
                if beam == goal_board:
                    solution = restore_solution(beam)
                    if not detail_output:
                        print_solution(solution, detail_output, 3)
                        return True
                    else:
                        print_kbeam(beam, explored, solution)
                        return True
                beam.find_neighbors()
                for neighbor in beam.neighbors:
                    neighbor.calcToGoal(goal_board)  # calculate heuristic to each child
                    neighbor.add_node_as_father(beam)  # set current node as father of each child
                all_neighbors.extend(beam.neighbors)
    print("No path found.")
    return False


#-----------------      GENETIC ALGHORITHM      ----------------------

#mutate of a child
def mutate(child, goal_board):
    child.find_neighbors()
    if child.neighbors:
        new_mutate = child.neighbors[random.randrange(0, len(child.neighbors))]
    else:
        new_mutate = child
    return new_mutate

#creating a child
def reproduce(x,y, goal_board):
    random_num = random.random()
    raffle = int(random_num*5) # a random int between 0-5
    if raffle == 0:
        reproduce = [x.myBoard[0], y.myBoard[1], y.myBoard[2], y.myBoard[3], y.myBoard[4], y.myBoard[5]]
    elif raffle == 1:
        reproduce = [x.myBoard[0], x.myBoard[1], y.myBoard[2], y.myBoard[3], y.myBoard[4], y.myBoard[5]]
    elif raffle == 2:
        reproduce = [x.myBoard[0], x.myBoard[1], x.myBoard[2], y.myBoard[3], y.myBoard[4], y.myBoard[5]]
    elif raffle == 3:
        reproduce = [x.myBoard[0], x.myBoard[1], x.myBoard[2], x.myBoard[3], y.myBoard[4], y.myBoard[5]]
    elif raffle == 4 or raffle == 5:
        reproduce = [x.myBoard[0], x.myBoard[1], x.myBoard[2], x.myBoard[3], x.myBoard[4], y.myBoard[5]]
    new_child = Board(reproduce, 0)
    if new_child.numOfAgents > 0:
        new_child.calcToGoal(goal_board)
    else:
        new_child.h_n = 100 # high score to decrease the probability to take the child
    return new_child


def random_selection(population):
    temp_probs = [] #list of the 1/h(n) of each node in population
    real_probs = [] #list of probabilities to take each node in population
    sum = 0
    for i, board in enumerate(population):
        if board.h_n == 0:
            switched_h = 1
        else:
            switched_h = 1/board.h_n
        temp_probs.append(switched_h)
        sum += switched_h
    for i in range(len(temp_probs)): #create a cumulative distribution function
        population[i].genetic_prob = temp_probs[i] / sum
        if i == 0:
            real_probs.append(population[i].genetic_prob)
        elif i == len(temp_probs) - 1:
            real_probs.append(1)
        else:
            real_probs.append(real_probs[i-1] + temp_probs[i] / sum)
    random_num = random.random()
    for i,board in enumerate(population): #check the chosen node in population
        if random_num <= real_probs[i]:
            return board


def create_population(population, sorted_neighbors):
    length = min(population,len(sorted_neighbors))
    pop = []
    for i in range(length):
        pop.append(sorted_neighbors[i])
    return pop


def genetic(population_size, starting_board,goal_board,detail_output):
    all_boards = []
    current = starting_board
    current.find_neighbors()
    parents_to_print = True
    mutated = False
    gen_population = create_population(population_size, current.neighbors)
    all_boards.append(starting_board)
    for node in gen_population:
        node.calcToGoal(goal_board)  # calculate heuristic to each node
    gen_population.sort(key=lambda x: x.h_n)
    for i in range(5000):
        new_population = []
        for i in range(population_size):
            x = random_selection(gen_population)
            y = random_selection(gen_population)
            if parents_to_print:
                parents_to_print = False
                parent1_to_print = x
                parent2_to_print = x
            if x == goal_board:
                print_gen_solution(parent1_to_print, parent2_to_print, mutated, detail_output, x, all_boards)
                return True
            if y == goal_board:
                print_gen_solution(parent1_to_print, parent2_to_print, mutated, detail_output, y, all_boards)
                return True
            child = reproduce(x,y, goal_board)
            if (random.random()<0.15):
                mutated = True
                child = mutate(child, goal_board)
            child.parent1_genetic = x
            child.parent2_genetic = y
            all_boards.append(child)
            if child == goal_board:
                print_gen_solution(parent1_to_print, parent2_to_print, mutated, detail_output, child, all_boards)
                return True
            new_population.append(child)
            new_population.sort(key=lambda x: x.h_n)
            gen_population = new_population
    print('No Path Found')
    return False



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
def print_solution(solution, detail_output, source):
    if detail_output:
        if source == 1:
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
        elif source == 2:
            for i in range(len(solution)):
                board = solution[i]
                if i == 0:
                    print("Board 1 (starting position):")
                    print()
                    print_current_board(board.myBoard)
                    print()
                    print_actions(board)
                    print("-----------")
                    print()
                elif i == len(solution) - 1:
                    print('Board ' + str(i + 1) + " (goal position):")
                    print_current_board(board.myBoard)
                else:
                    print('Board ' +str (i+1) + ":")
                    print_current_board(board.myBoard)
                    print()
                    print_actions(board)
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

#print solution for genetic
def print_gen_solution(parent1, parent2, mutated, detail_output, child, all_boards_list):
    if detail_output:
            print()
            print("Starting board 1: (probability of selection from population::<", parent1.genetic_prob, ">):")
            print_current_board(parent1.myBoard)
            print()
            print("Starting board 2: (probability of selection from population::<", parent2.genetic_prob, ">):")
            print_current_board(parent2.myBoard)
            print()
            if mutated == True:
                string_to_print = "yes"
            else:
                string_to_print = "no"
            print("Result board: (mutation happend::<", string_to_print, ">):")
            print_current_board(child.myBoard)
    else:
        abc = 'abc'
        iter = 2
        for i, board in enumerate(all_boards_list):
            char = 0
            if board != child:
                if i == 0:
                    print("Board 1 (starting position):")
                    print_current_board(board.myBoard)
                    print("-----------")
                    print()
                    print("Parent 2a:")
                    print_current_board(parent1.myBoard)
                    print()
                    print("Parent 2b:")
                    print_current_board(parent2.myBoard)
                    print()
                else:
                    print('Board ' + str(i + 1) + ":")
                    print_current_board(board.myBoard)
                    print("-----------")
                    print()
                    print('Board ' + str(iter) + 'a:')
                    print_current_board(board.parent1_genetic.myBoard)
                    print("-----------")
                    print()
                    print('Board ' + str(iter) + 'b:')
                    print_current_board(board.parent1_genetic.myBoard)
                    print("-----------")
                    print()
                    char += 1
                    iter += 1
            else:
                print('Board ' + str(i+1) + " (goal position):")
                print_current_board(board.myBoard)


#print solution for k-beam
def print_kbeam(beam, explored, solution):
    abc = 'abc'
    iter = 2
    for i, board in enumerate(solution):
        char = 0
        if board != beam:
            if i == 0:
                print("Board 1 (starting position):")
                print_current_board(board.myBoard)
                print("-----------")
                print()
            else:
                print('Board ' + str(i + 1) + ":")
                print_current_board(board.myBoard)
                print("-----------")
                print()
            for visited in explored:
                if visited.father == board:
                    print('Board ' + str (iter) + abc[char] + ':')
                    print_current_board(visited.myBoard)
                    print("-----------")
                    print()
                    char += 1
            iter += 1
        else:
            print('Board ' + str(len(solution)) + " (goal position):")
            print_current_board(board.myBoard)



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

def print_actions(board):
    for i in range(len(board.considered_actions)):
        print("action:",board.considered_actions[i],"; probability: " , board.probabilities[i])




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
        self.restore_move = None
        self.considered_actions = [] #considered actions for simulated annealing algo
        self.probabilities = [] #probabilities for simulated annealing algo
        self.genetic_prob = 0 # probability for genetic algo
        self.parent1_genetic = None #parent1 for genetic algo
        self. parent2_genetic = None #parent2 for genetic algo


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
        # if countAgent == 0:
        #     return 0
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
            temp_board.restore_move = "("+str(i+1)+","+str(j+1)+")->("+str(i+1)+","+str(j+2)+")"
            self.neighbors.append(temp_board)
        if source == 2:
            temp_board.myBoard[i][j] = 0
            temp_board.myBoard[i][j-1] = 2
            temp_board.restore_move = "("+str(i+1)+","+str(j+1)+")->("+str(i+1)+"," +str(j) + ")"
            self.neighbors.append(temp_board)
        if source == 3:
            temp_board.myBoard[i][j] = 0
            temp_board.myBoard[i-1][j] = 2
            temp_board.restore_move = "("+str(i+1)+","+str(j+1)+")->(" + str(i) + "," + str(j+1) + ")"
            self.neighbors.append(temp_board)
        if source == 4:
            if i == self.numOfRow - 1:
                temp_board.myBoard[i][j] = 0
                temp_board.restore_move = "("+str(i+1)+","+str(j+1)+")-> outside the board"
            else:
                temp_board.myBoard[i][j] = 0
                temp_board.myBoard[i+1][j] = 2
                temp_board.restore_move = "("+str(i+1)+","+str(j+1)+")->(" + str(i + 2) + "," + str(j+1) + ")"
            self.neighbors.append(temp_board)

    #count number of agents on board
    def check_num_of_agents(self):
        agentCounter = 0
        if self.numOfRow != 6 or self.numOfCol != 6:
            return 0
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
    starting_board = Board(starting_board, 0)
    goal_board = Board(goal_board, 0)
    if check_valid(starting_board, goal_board):
        if search_method == 1:
            solve = Astars(starting_board, goal_board, detail_output)
        elif search_method == 2:
            solve = hill_climbing(starting_board, goal_board, detail_output)
        elif search_method == 3:
            solve = simulated_annealing(starting_board, goal_board, detail_output)
        elif search_method == 4:
            solve = k_beam(3, starting_board, goal_board, detail_output)
        elif search_method == 5:
            solve = genetic(10, starting_board, goal_board, detail_output)
    else:
        print("Warning: unvalid boards")


if __name__ == "__main__":
    find_path((
            [[2, 0, 2, 0, 2, 0],
             [0, 0, 0, 2, 1, 2],
             [1, 0, 0, 0, 0, 0],
             [0, 0, 1, 0, 1, 0],
             [2, 0, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, 0]]), (
            [[2, 0, 2, 0, 0, 0],
             [0, 0, 0, 2, 1, 2],
             [1, 0, 0, 0, 0, 2],
             [0, 0, 1, 0, 1, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, 0]]), 5, True)