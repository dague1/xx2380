#!/usr/bin/env python3
#from _typeshed import NoneType
from bauch import Bauch
from alpha_pass_v2 import Alpha
from player_controller_hmm import PlayerControllerHMMAbstract
from constants import *
#import random
class PlayerControllerHMM(PlayerControllerHMMAbstract):
    def init_parameters(self):
        #uniform at first *worked* but should preferrably be random
        self.O, self.O_t, self.species, self.fish  = [], [], [], -1
        self.A = [[[1 / 5 for i in range(5)] for j in range(5)]  for z in range(N_FISH)] #3d varje element här är en lista med 5 st listor:[0.2, 0.2, 0.2, 0.2, 0.2], så A [[[0.2, 0.2, 0.2, 0.2, 0.2],....
        #print('A',self.A)
        self.B = [[[1 / N_EMISSIONS for i in range(N_EMISSIONS)] for j in range(5)] for k in range(N_FISH)] # -II- [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125] (8 emissions)
        #print('B',self.B)
        self.pi = [[1 / 5 for i in range(5)] for j in range(N_FISH)] # varje element [[0.2, 0.2, 0.2, 0.2, 0.2].. 2dlista
        #print('pi', self.pi)
        #initialize the species, each specie will be represented by an HMM model (7)
        for i in range(N_SPECIES):
            self.species.append(0)
        pass

    def ideal_guess(self):
        curr_best, alpha_pass_instance,curr_best_index, tuple = 0, Alpha(), 0, zip(self.species, range(N_SPECIES))
        #print('tuple', list(tuple)) tuple [(0, 0), (0, 1), (0, 2), (0, 3), ([[[0.2000000000000005, 0.2000000000000005,....]]]))
        # Performs forward algorithm on the 7 models going from fish 70 to 1
        for model, i in tuple:
            if model == 0:
                continue
            else:
                probability = alpha_pass_instance.alpha_pass_algorithm_v2(model[0], model[1], model[2], self.O_t[self.fish])
                if probability > curr_best:
                    curr_best = probability
                    curr_best_index = i
        return curr_best_index
    #returns true if specie
    def empty(self):
        bool = True
        for m in self.species:
            if m == 0:
                continue
            else:
                bool = False
        return bool
    """
        This method gets called on every iteration *even before we get score*, providing observations.
        Here the player should process and store this information,
        and optionally make a guess by returning a tuple containing the fish index and the guess.
        :param step: iteration number
        :param observations: a list of N_FISH observations, encoded as integers
        :return: None or a tuple (fish_id, fish_type)
        """
    def guess(self, step, observations):
        print(step)
       # print('guess has been called')
        if step >= 100: #step is the iteration number of player loop in player_controller_hmm.
            # Higher step threshold gives more observations and more accurate baum welch results. 
            # this number should be high enough to yield many observations, but low enough to not timeout (90s) before finishing guesses. 
            #print(observations)
            self.O.append(observations)
            #print(self.O)
            #as we get one list of 70 observations, and each observation is of a fish_ID, ordered it seems, if we list map list zip * we will 
            #make a new list with each emission for each fish, which is what we want for our model.
            #O_t will be the emissions for each fish ID, so 
            #(an observation is just a digit between 0 and 8 and represents some kind of movement)
            self.O_t = list(map(list, zip(*self.O))) # list1 = [[1,2,3], [4,5,6], [7,8,9]] will give list 2 = [[1, 4, 7], [2, 5, 8], [3, 6, 9]]
            #print(self.O_t)
            #
            self.fish = self.fish + 1
            if(self.empty()!=False):
                return (self.fish,0)#random gissning h
            else:
                bestspecies = self.ideal_guess()
                #print(self.species)
                return (self.fish, bestspecies)
        else:
            self.O.append(observations)
            #print(self.species)
        return None
    """
        (really called when we get scores. When we start revealing)
        This methods gets called whenever a guess was made.
        It informs the player about the guess result
        and reveals the correct type of that fish.
        :param correct: tells if the guess was correct
        :param fish_id: fish's index
        :param true_type: the correct type of the fish
        :return:
        Called in player_controller_hmm
        """
        #
        #species a list of 7 elements. First all 0s, then each specie is given an Ax, Bx, pix value. 
    def reveal(self, correct, fish_id, true_type):
        #print('reveal has been called')
        if(correct is not True): #goes in here every time we make wrong guess. (12 times if we get score 58 etc)
            model = Bauch()
            #print(fish_id)
            #print(self.O_t)
           # print('O', self.O_t)
           # print('O_T', self.O_t)
            Ax, Bx, pix = model.model_loop(self.A[fish_id], self.B[fish_id], self.pi[fish_id], self.O_t[fish_id]) #
            self.species[true_type] = [Ax, Bx, pix] # Sets the fish specie of index true_type to the Baum Welch result. 
            #print(self.species)
        pass
