import sys
out = sys.stdout


#The SECOND HMM problem. What sequence best explains a sequence of observations?
# Strat 1: choose states that are individually most likely. Problem? this strategy could find states that are not legal according to our markov model. 

#Receives a string-input as specified in Kattis, and makes a matrix out of it.
def generate_matrix(line):
    line = line.split()
    floatLine = [float(i) for i in line]
    rows = int(floatLine[0])
    cols = int(floatLine[1])
    matrix = []
    for i in range(rows):
        row = []
        for j in range(cols):
            row.append(floatLine[j + i * cols + 2]) # + 2 cuz we skip first 2 dimension elements
        matrix.append(row)
    return matrix

def generate_probability_sequence(line):
    line = line.split()
    floatLine = [float(i) for i in line]
    prob_sequence = []
    num = int(floatLine[0])
    for i in range(num):
        prob_sequence.append(int(line[i + 1]))

    #print(prob_sequence)    
    return prob_sequence

#In the hidden Markov model we use two matrices. The first one, called the transition matrix, 
# determines probabilities of transitions from one hidden state to another one (the next one). 
# The second matrix, called the emission matrix, determines probabilities of observations given a hidden state.

#In other words, we can imagine a system as being in a state (which is hidden, unobservable) 
# and this hidden state determines the probability of the next hidden state as well as probability of a given observation.

#It means that we assume that a "jump" (or transition) to the next hidden state and "generation" of a certain observation are independent events. 
# This IS a LIMITATION of the model
#
#### yay not working with classes here
## A, B and p are the input matrices
# What is the path with the highest probability that accounts for the first t observations and ends at state Si?

#
#Very similar to alpha beta algorithm, but instead of calculating the sum of all previous probabilities,
#  we just want the MAX of all the previous probabilities. 
#
def modified_viterbi(A, B, p, O):
    N, T, d, didx = len(A), len(O), [], []
    #initialize delta-matrix. T number of rows.
    for i in range(0, T):
        d.append([])
        #initialize deltaIDX-matrix. T-1 number of rows.
    for i in range(1, T):
        didx.append([])    
    #initiate values of first row of delta 
    for i in range(0, N):
        d[0].append(B[i][O[0]]*p[0][i])
    # generate /calc the rest of the d-matrix
    # go through each obs in the seq
    for i in range(1, T):
        #and the current d-matrix row
        for j in range(0, N):
            arr = []
            #go through possible seq of states
            for k in range(0, N):
                arr.append(A[k][j]*d[i-1][k]*B[j][O[i]]) #(2.21)
           #add largest value to d
            max_val = max(arr)
            d[i].append(max_val)
            didx[i-1].append(arr.index(max_val)) # we need to keep track of which i maximized the result at each time step.
    #Backtracking step (step 2)
    res = backtrack(d, didx, T)
    return res
    
#Backtracking step 
#calculate the most probable sequence
#starts 
#res [1]
# didx [[0, 0, 0, 0], [0, 0, 1, 0], [0, 2, 0, 0]]
#res [1, 2]
# didx[[0, 0, 0, 0], [0, 0, 1, 0]]
# res[1, 2, 1]
# didx [[0, 0, 0, 0]]


def backtrack(d, didx, T):
    res = []
    res.append(d[T-1].index(max(d[T-1])))
    for i in range(1, T):
        res.append(didx[-1][res[i-1]])
        del didx[-1]
    res.reverse()
    return res  

line1 = sys.stdin.readline() #transition matrix
line2 = sys.stdin.readline() #emission matrix
line3 = sys.stdin.readline() #initial state probability distribution
line4 = sys.stdin.readline() #number of emissions and the sequence of emissions itself (we want to calculate the probability of seq ( 0 1 2 3 0 1 2 3 ) given (8 0 1 2 3 0 1 2 3) 
A = generate_matrix(line1)
B = generate_matrix(line2)
p = generate_matrix(line3)
seq = generate_probability_sequence(line4)
sequence = modified_viterbi(A,B,p,seq)
print(*sequence, sep=" ")

