import sys
out = sys.stdout

#!!!!!calculate the probability to observe a certain emission sequence given a HMM model!!
#
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
    #print(matrix)    
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

def matmul(A,B):
    r=[]
    mat=[]
    for i in range(len(A)):
        for j in range(len(B[0])):
            sum=0
            for k in range(len(B)):
                sum=sum+(A[i][k]*B[k][j])
            r.append(round(sum,3))
        mat.append(r)
        r=[]
    return mat


def format_output(matrix):
    outputStr = ""
    outputStr += str(len(matrix)) + " " + str(len(matrix[0]))
    for i in matrix: 
        for j in i:
            outputStr += " " + str(j)
    return outputStr
#O(nm^2) where n is length of seq, m is number of hidden states
#https://www.youtube.com/watch?v=9-sPm4CfcD0
def alpha_pass_algorithm(A, B, pi, O):
    arr = []
    for i in range(0, len(A)):
        arr.append(B[i][O[0]]*pi[0][i]) #  (2.8)We start off by computing the probability of having observed the first observation o1 and having been in any of the hidden states
    #print(templist)
    alpha = arr
    #we loop over all states
    for t in range(0, len(O)-1):
        arr = []
        for i in range(0, len(A)):
            sigma = 0
            for j in range(0, len(A)):
                sigma += alpha[j]*A[j][i]
            arr.append(B[i][O[t+1]]*sigma) #2.13 in assignment
        alpha = arr
    probability = sum(alpha)    
    return probability



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
line1 = sys.stdin.readline() #transition matrix
line2 = sys.stdin.readline() #emission matrix
line3 = sys.stdin.readline() #initial state probability distribution
line4 = sys.stdin.readline() #number of emissions and the sequence of emissions itself (we want to calculate the probability of seq ( 0 1 2 3 0 1 2 3 ) given (8 0 1 2 3 0 1 2 3) 
A = generate_matrix(line1)
B = generate_matrix(line2)
p = generate_matrix(line3)
seq = generate_probability_sequence(line4)
#A,B,p = get_input(sys.stdin)
#p_A = matmul(p, A) 
#p_A_B = matmul(p_A, B)
print(alpha_pass_algorithm(A, B, p, seq))

#out.write(x)