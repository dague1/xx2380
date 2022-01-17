#!/usr/bin/env python3
from fishing_game_core.game_tree import Node
from fishing_game_core.player_utils import PlayerController
from fishing_game_core.shared import ACTION_TO_STR

import numpy as np
import random
import math
import time 

#Kevin & David
class PlayerControllerHuman(PlayerController):
    def player_loop(self):

        while True:
            # send message to game that you are ready
            msg = self.receiver()
            if msg["game_over"]:
                return


class PlayerControllerMinimax(PlayerController):

    def __init__(self):
        super(PlayerControllerMinimax, self).__init__()

    def player_loop(self):
        # Generate game tree object
        first_msg = self.receiver()
        
        self.generateZobristMatrix(len(first_msg)-1)
        while True:
            msg = self.receiver()

            #self.visited = {}
            # Create the root node of the game tree
            node = Node(message=msg, player=0)

            # Possible next moves: "stay", "left", "right", "up", "down"
            best_move = self.search_best_next_move(initial_tree_node=node)

            # Execute next action
            self.sender({"action": best_move, "search_time": None})

    def search_best_next_move(self, initial_tree_node):
        return ACTION_TO_STR[self.runModel(initial_tree_node)]

  # Runs alpha beta pruning with iterative deepening search using a time limit
    def runModel(self, node):
        startingTime, errorThrown, depth, visitedNodes, idealmove = time.time(), False, 0, dict(), 0
        while not errorThrown:
            try:
                nextMove = self.iterativeDeepening(node, depth, startingTime, visitedNodes)
                idealMove = nextMove
                depth += 1
            except:
                errorThrown = True
                break
        return idealMove 

  # Iterative deepening search (incrementation of depth done in runModel)
    def iterativeDeepening(self, node, depth, startingTime, visitedNodes):
        a, b, childNodes, listOfScores = float('-inf'), float('inf'), node.compute_and_get_children(), []
        #gå igenom childnodesen och lägg in alla v-värden i en lista
        for child in childNodes:
            v = self.alphabetaMinimax(child.state, child, depth, a, b, 1, startingTime, visitedNodes)
            listOfScores.append(v)
        #hitta det maximala scoret i listOfScores och returnera indexet av den childnoden.
        indexOfBest = listOfScores.index(max(listOfScores))
        return childNodes[indexOfBest].move # ACTION_TO_STR kommer göra om movet till en sträng tex LEFT. .move är en attribute, 
        #
        #ACTION_TO_STR = {
        #0: "stay",
        #1: "up",
        #2: "down",
        #3: "left",
        #4: "right"
        #}





    #visitedNodes contains
    # {"{0: (5, 12), 1: (11, 17)}{'514': 11, '1914': 2, '1016': 10, '912': 2, '1811': 11}": [0, 10.9978]}
    # ett state enl vår hash pekar på en lista med [depth, v]
    #
    def alphabetaMinimax(self, state, node, depth, a, b, player, startingTime, visitedNodes):
        if  0.052 < time.time() - startingTime:
            raise TimeoutError
        else:
            childNodes = node.compute_and_get_children()
            #print(visitedNodes)
            key = self.generateZobristHash(state) #generate zobrist hashkey for the state
            #check for repeated states
            if key in visitedNodes: #vistedNodes är en dict  (hashtable). om nyckeln finns( vi har besökt noden),
                if visitedNodes[key][0] >= depth: #och den är djupare än current depth (ett djupare state är senare i tiden och därför prio)
                    return visitedNodes[key][1] # returnera dess v-värde istället för att gå ner rekursivt till något som leder dit
            

            if depth == 0 or len(childNodes) == 0:
                v = self.taxiHeuristics(node) # if we reach a terminal state, evaluate the utility according to heuristics
            elif player == 0:
                v = -999999
                for c in childNodes:
                    v = max(v, self.alphabetaMinimax(c.state, c, depth - 1, a, b, 1, startingTime,visitedNodes))
                    a = max(a, v)
                    if a >= b:
                        break
            else:
                v = 999999
                for c in childNodes:
                    v = min(v, self.alphabetaMinimax(c.state, c, depth - 1, a, b, 0, startingTime,visitedNodes))
                    b = min(b, v)
                    if b <= a:
                        break
            visitedNodes.update({key:[depth,v]}) 
        #print(v)    
        return v


    #Heuristics using taxi distance. Maximize priority if hook is on top of fish.  
    #Else; Prioritize the closest fish in a linear fasion. 
    # when dist gets closer to 0, priorizedValue gets closer to simply the fish score.
    def taxiHeuristics(self, node):
        prioritizedValue = 0
        for fish in node.state.fish_positions: #loop over fishes (we want to decide which one is best for us)
            dist = self.taxiDistance(node.state.fish_positions[fish], node.state.hook_positions[0]) # cab dist between p0 hook and fish
            if dist == 0 and node.state.fish_scores[fish] > 0: #if hook is on top of fish, maximize heuristic value
                return 999999
            prioritizedValue = max(prioritizedValue, node.state.fish_scores[fish] * self.f(dist)) #else, prioritize close fish according to the straight line y=-0.0001*dist+1. 
 
        return self.utilityOfState(node) + prioritizedValue  #return the current score difference between green and red boat, plus what we would gain from prioritizedValue

 # Straight line used for the heuristics
    def f(self, x):
        k = -0.00001
        m = 1
        y = k*x+m
        return y
    
    # Returns the utility of a state; that is, the difference green boat score - red boat score of a given state. 
    def utilityOfState(self, node):
        gamma = node.state.player_scores[0] - node.state.player_scores[1]
        return gamma



    #Returns Cab-distance between hook and fish.
    #
    # + - - -
    #       |
    #       |
    #       O  
    # Distance here would be 5 (not Euclidean)
    #  Also considers the distance if the hook crosses the x axis and comes out on the other side of the screen.
    def taxiDistance(self, hookPos, fishPos):
        return min(20 - abs(hookPos[0] - fishPos[0]), abs(hookPos[0] - fishPos[0])) + abs(hookPos[1] - fishPos[1])


  
#
# sets up Zobrist table
# 400 is the size of map,  20*20.
# 2 + numberOfFish is 7 for default map as there are 5 fish and 2 hooks.
# initiate a matrix [number of "board cells"number of pieces][number of pieces] mat[20*20][7] here
    def generateZobristMatrix(self,numberOfFish):
        self.zob_table = [[random.randrange(2**64-1) for i in range(2 + numberOfFish)] for j in range(400)]
        #print(self.zob_table)
        #print(len(self.zob_table))
        

#
# Zobrist hash from https://iq.opengenus.org/zobrist-hashing-game-theory/
#returns a hash according to current board configuration
# as we map a 2d-matrix (zobtable) to a 1-d list (hookPos and fishPos),  position y x in matrix will have position y*20+x in list. hence we multiply y coord with 20
# 400 rows, every 20th row is a new y-coord in zobtable so y*20+x can access all elements
    def generateZobristHash(self, state):
        hashVal, p0Hook = 0, state.get_hook_positions()[0]
        for fish, fishPos in state.get_fish_positions().items(): #dict_items([(0, (5, 14)), (1, (19, 14)), (2, (10, 16)), (3, (9, 12)), (4, (18, 11))])
           # print(state.get_fish_positions().items())
            hashVal ^= self.zob_table[fishPos[1]*20 + fishPos[0]][fish] #xora hashvärdet med zob_table[y koord_fisk*20 + x koord_fisk][fiskens index]   
        
        for player, hookPos in state.get_hook_positions().items():
            hashVal ^= self.zob_table[hookPos[1]*20 + hookPos[0]][player] #xora hashvärdet med zob_table[y koord_hook*20 + x koord_hook][spelarens index]
        return hashVal


  

  