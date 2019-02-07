# pacmanAgents.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

from pacman import Directions
from game import Agent
from heuristics import *
import random

class RandomAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # get all legal actions for pacman
        actions = state.getLegalPacmanActions()
        # returns random action from all the valide actions
        return actions[random.randint(0,len(actions)-1)]

class OneStepLookAheadAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # get all legal actions for pacman
        legal = state.getLegalPacmanActions()
        # get all the successor state for these actions
        successors = [(state.generatePacmanSuccessor(action), action) for action in legal]
        # evaluate the successor states using scoreEvaluation heuristic
        scored = [(admissibleHeuristic(state), action) for state, action in successors]
        # get best choice
        bestScore = min(scored)[0]
        # get all actions that lead to the highest score
        bestActions = [pair[1] for pair in scored if pair[0] == bestScore]
        # return random action from the list of the best actions
        return random.choice(bestActions)

class BFSAgent(Agent):
    
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        visit_state = []
        visited={}
        prv={}
        depth={}
        depth[state]=0
        start_node = state
        visit_state.append((state,Directions.STOP))
        
        while visit_state:
            curr_node = visit_state[0]
            del visit_state[0]
            
            if curr_node[0].isWin():
                break
            
            legal = curr_node[0].getLegalPacmanActions()
            # get all the successor state for these actions
            successors = [(curr_node[0].generatePacmanSuccessor(action),action) for action in legal]
            for successor in successors:
                if successor[0]==None:
                    break
                if successor[0] in visited:
                    continue
                visited[successor[0]]=True
                depth[successor[0]] = depth[curr_node[0]] + 1
                visit_state.append(successor)
                prv[successor[0]] = (curr_node[0],successor[1])

                
        path = []
        minimum=999999999
        for curr_node in prv:
            if admissibleHeuristic(curr_node)+depth[curr_node]<minimum:
                minimum=admissibleHeuristic(curr_node)+depth[curr_node]
                finalnode = curr_node
                
        
        curr_node = finalnode
        while curr_node != start_node:
            path.append((prv[curr_node][1],prv[curr_node][0]))
            curr_node = prv[curr_node][0]
            
        soln = path[::-1]
        # print(admissibleHeuristic(soln[1][1]))

        # TODO: write BFS Algorithm instead of returning Directions.STOP
        return soln[0][0]

class DFSAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        visit_state = []
        visited={}
        prv={}
        depth={}
        depth[state]=0
        start_node = state
        visit_state.append((state,Directions.STOP))
        count=0
        
        while visit_state:
            count+=1
            curr_node = visit_state[-1]
            del visit_state[-1]
            minimum = admissibleHeuristic(curr_node[0])
            
            if curr_node[0].isWin():
                break
            
            legal = curr_node[0].getLegalPacmanActions()
            # get all the successor state for these actions
            successors = [(curr_node[0].generatePacmanSuccessor(action),action) for action in legal]
            for successor in successors:
                if successor[0]==None:
                    break
                if successor[0] in visited:
                    continue
                depth[successor[0]] = depth[curr_node[0]] + 1
#                 elif admissibleHeuristic(successor[0])<minimum or visit_state == []:
                visited[successor[0]]=True
                visit_state.append(curr_node)
                visit_state.append(successor)
                prv[successor[0]] = (curr_node[0],successor[1])
#                 break

                
        path = []
        minimum=999999999
        for curr_node in prv:
            if admissibleHeuristic(curr_node)+depth[curr_node]<minimum:
                minimum=admissibleHeuristic(curr_node)+depth[curr_node]
                finalnode = curr_node
                
        
        curr_node = finalnode
        while curr_node != start_node:
            path.append((prv[curr_node][1],prv[curr_node][0]))
            curr_node = prv[curr_node][0]
            
        soln = path[::-1]
#         print(admissibleHeuristic(soln[1][1]))
        # TODO: write BFS Algorithm instead of returning Directions.STOP
        return soln[0][0]
    

class AStarAgent(Agent):
    
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        
        total_costs=[]
        visited={}
        prv={}
        cost_dict = {}
        depth={}
        start_node = state
        
        cost_dict[state]=0
        depth[state]=0
        total_costs.append((state,0))
        min_state = state
        min_cost = 999999999 + admissibleHeuristic(state)
        while total_costs:
            curr_node = sorted(total_costs,key=lambda x: x[1], reverse=True)[0][0]
            total_costs = sorted(total_costs,key=lambda x: x[1], reverse=True)
            del total_costs[0]
            g_x_curr_node = cost_dict[curr_node]
            if curr_node.isWin():
                break
            if curr_node in visited:
                continue
            visited[curr_node]=True

            legal = curr_node.getLegalPacmanActions()

            successors = [(curr_node.generatePacmanSuccessor(action),action) for action in legal]
            for successor in successors:
                if successor[0]==None:
                    break
                depth[successor[0]] = depth[curr_node] + 1
                g_x_successor = g_x_curr_node + depth[successor[0]]
                f_x = g_x_successor + admissibleHeuristic(successor[0])
                if cost_dict.has_key(successor[0])==False or g_x_successor < cost_dict[successor[0]]: 
                    cost_dict[successor[0]] = g_x_successor
                    total_costs.append((successor[0],f_x))
                    prv[successor[0]] = (curr_node,successor[1])
                if f_x <min_cost:
                    min_state=successor[0]
                    min_cost=f_x

                
        path = []                
        curr_node = min_state
        while curr_node != start_node:
            path.append((prv[curr_node][1],prv[curr_node][0]))
            curr_node = prv[curr_node][0]
        soln = path[::-1]
        # print(admissibleHeuristic(soln[0][0]))
        # TODO: write A* Algorithm instead of returning Directions.STOP
        return soln[0][0]
		
class RandomSequenceAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        self.actionList = [];
        for i in range(0,10):
            self.actionList.append(Directions.STOP);
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # get all legal actions for pacman
        possible = state.getAllPossibleActions();
        for i in range(0,len(self.actionList)):
            self.actionList[i] = possible[random.randint(0,len(possible)-1)];
        tempState = state;
        for i in range(0,len(self.actionList)):
            if tempState.isWin() + tempState.isLose() == 0:
                tempState = tempState.generatePacmanSuccessor(self.actionList[i]);
            else:
                break;
        # returns random action from all the valide actions
        return self.actionList[0];

    
class HillClimberAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        self.actionList = [];
        for i in range(0,5):
            self.actionList.append(Directions.STOP);
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        
        tempState=currstate=state
        currmax = gameEvaluation(currstate,tempState)
        bestaction=self.actionList[0]
        firstitr=True
        flag=False
        loseflag=False
        possible = tempState.getAllPossibleActions();
        
        while True:
            tempState=state
            

            for i in range(0,len(self.actionList)):
                ''' For the first action sequence we dont need to change the action'''
                if firstitr:
                    self.actionList[i] = possible[random.randint(0,len(possible)-1)]
                    firstitr=False
                else: 
                    ''' Here we change the action, probability of changing=0.5'''
                    if random.randint(0,1)==1:
                        self.actionList[i] = possible[random.randint(0,len(possible)-1)]
            
            for i in range(0,len(self.actionList)):
                tempState = tempState.generatePacmanSuccessor(self.actionList[i])
                if tempState==None:
                    flag=True
                    break
                if tempState.isLose():
                    loseflag=True
                    break
                elif tempState.isWin():
                    return self.actionList[0]
            
            if flag:
                break
            if loseflag:
                loseflag=False
                continue
            if gameEvaluation(currstate,tempState)>currmax:
                currmax=gameEvaluation(currstate,tempState)
                bestaction=self.actionList[0]

        return bestaction
        
    
class GeneticAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        self.actionList = [];
        for i in range(0,5):
            self.actionList.append(Directions.STOP);
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # TODO: write Genetic Algorithm instead of returning Directions.STOP
        populationList = []
        tempState=currstate=state
        currmax = gameEvaluation(currstate,tempState)
        bestaction=self.actionList[0]
        possible = tempState.getAllPossibleActions();
        flag=True
        loseflag=False
        
        for i in range(0,8):
            for j in range(0, len(self.actionList)):
                self.actionList[j] = possible[random.randint(0, len(possible) - 1)]
            populationList.append(self.actionList[:])

        while flag:
            population = []
            for i in range(0,8):
                tempState=state
                for j in range(0, len(self.actionList)):
                    tempState = tempState.generatePacmanSuccessor(populationList[i][j])
                    if tempState==None:
                        flag=False
                        break
                    if tempState.isLose():
                        break
                    elif tempState.isWin():
                        return self.actionList[0]
                if flag==False:
                    break
                population.append((populationList[i][:],gameEvaluation(currstate,tempState)))
            
            if flag==False:
                break
            population.sort(key = lambda x:x[1], reverse=True)
            if population[0][1]>currmax:
                bestaction=population[0][0][0]
            
            populationList=[]
            for i in range (0,4):
                rankpart1,rankpart2 = self.pairSelect()
                x_chromosome = population[rankpart1][0][:]
                y_chromosome = population[rankpart2][0][:]
                crossover = random.randint(0,100)
                
                if random.randint(0,100)<=70:
                    child1=[]
                    child2=[]
                    for j in range(0,5):
                        if random.randint(0,1)==1:
                            child1.append(x_chromosome[j])
                            child2.append(y_chromosome[j])
                        else:
                            child1.append(y_chromosome[j])
                            child2.append(x_chromosome[j])
                    populationList.append(child1)
                    populationList.append(child2)
                else:
                    populationList.append(x_chromosome[:])
                    populationList.append(y_chromosome[:])
                    
            for i in range(0,8): 
                if random.randint(0, 100) <= 10:
                    populationList[i][random.randint(0,4)] = possible[random.randint(0,len(possible)-1)]
                    
        return bestaction
    
    def pairSelect(self):
        x=random.randint(1,36)
        y=random.randint(1,36)
        
        x = self.calculateProbability(x)
        y = self.calculateProbability(y)
        ''' Here we go on selecting y until it differs from x.
        This is done in order to avoid same X and Y chromosome'''
        while x==y:
            y=random.randint(1,36)
            y = self.calculateProbability(y)
            
        return x,y
    
    def calculateProbability(self,number):
        
        if number==1:
            return 7
        elif number<=3:
            return 6
        elif number <=6:
            return 5
        elif number<=10:
            return 4
        elif number <=15:
            return 3
        elif number<=21:
            return 2
        elif number <=28:
            return 1
        elif number<=36:
            return 0



class MCTSAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        self.root = None
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # TODO: write MCTS Algorithm instead of returning Directions.STOP
        bestActions =[]
        rootChildList = []
        self.root = Node(None,None)
        
        while True:
            expandedNode = self.treePolicy(state)
            if expandedNode == None:
                break
            score = self.defaultPolicy(expandedNode, state)
            if score == None:
                break
            self.backUp(expandedNode, score)
            
        for node in self.root.childList:
            rootChildList.append(node.visitnumber)
            
        rootChildList.sort(reverse=True)  
        
        for node in self.root.childList:
            if node.visitnumber == rootChildList[0]:
                bestActions.append(node.prevaction)
        
        return bestActions[random.randint(0,len(bestActions)-1)]


    def backUp(self, node, score):
        while node is not self.root:
            node.visitnumber +=  1
            node.score += score
            node = node.prev
        

    def defaultPolicy(self, node, rootState):
        nodeSet = []
        tempNode = node
        while tempNode.prev is not None:
            nodeSet.append(tempNode)
            tempNode = tempNode.prev
        nodeSet.reverse()
        tempState = rootState
        for i in nodeSet:
            prevState = tempState
            tempState = tempState.generatePacmanSuccessor(i.prevaction)
            if tempState is None:
                self.backUp(i, gameEvaluation(rootState, prevState))
                return None
            elif tempState.isWin():
                self.backUp(i,gameEvaluation(rootState, tempState))
                return 
            elif tempState.isLose():
                self.backUp(i,gameEvaluation(rootState, tempState))
                return 


        for j in range(0,5):
            if tempState.isWin() + tempState.isLose() == 0:
                legal = tempState.getLegalPacmanActions()
                if not legal:
                    break
                prevState = tempState
                tempState = tempState.generatePacmanSuccessor(random.choice(legal))
                if tempState is None:
                    return None
            else:
                break
        return gameEvaluation(rootState, tempState)

    def bestChild(self, node):
        currmax = -999999
        bestChild = []                        
        for i in node.childList:
            result = (i.score/i.visitnumber) + 1*math.sqrt((2*math.log(node.visitnumber))/i.visitnumber)
            if result == currmax:
                bestChild.append(i)
            if result > currmax:
                currmax = result
                bestChild = []
                bestChild.append(i)
        return bestChild[random.randint(0, len(bestChild) - 1)]

    def expansion(self, node, rootState):
        nodeSet = actionList = []
        tempNode = node
        
        while tempNode.prev != None:
            nodeSet.append(tempNode)
            tempNode = tempNode.prev

        nodeSet.reverse()
        tempState = rootState

        for i in nodeSet:
            prevState = tempState
            tempState = tempState.generatePacmanSuccessor(i.prevaction)
            if tempState is None:
                self.backUp(i, gameEvaluation(rootState, prevState))
                return None
            elif tempState.isWin():
                self.backUp(i,gameEvaluation(rootState, tempState))
                return 
            elif tempState.isLose():
                self.backUp(i,gameEvaluation(rootState, tempState))
                return 
            
        for i in node.childList:
            actionList.append(i.prevaction)
        
        legal = tempState.getLegalPacmanActions()
        for action in legal:
            if action not in actionList:            
                childNode = node.createChild(action)
                break
        if len(node.childList) == len(legal):                
            node.expanded = True
        return childNode


    def treePolicy(self, root):
        node = self.root
        while True:
            if not node.expanded:
                return self.expansion(node, root)
            else:
                node = self.bestChild(node)


class Node:
    score = visitnumber = 0
    childList = []
    expanded = False
    prevaction = None
    
    def __init__(self, prevaction, prev):
        self.score = 0
        self.childList = []
        self.visitnumber = 1
        self.expanded = False
        self.prevaction = prevaction
        self.prev = prev

    def createChild(self, action):
        newNode = Node(action, self)
        self.childList.append(newNode)
        return newNode