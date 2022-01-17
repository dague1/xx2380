import math
import sys


out = sys.stdout
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

def generate_emission_list(line):
    line = line.split()
    floatLine = [float(i) for i in line]
    observations = []
    num = int(floatLine[0])
    for i in range(num):
        observations.append(int(line[i + 1]))  
    return observations

def format_output(matrix):
    outputStr = ""
    outputStr += str(len(matrix)) + " " + str(len(matrix[0]))
    for i in matrix: 
        for j in i:
            outputStr +=  " " + str(j)
    return outputStr

#O(nm^2) where n is length of seq, m is number of hidden states
#https://www.youtube.com/watch?v=9-sPm4CfcD0
def alpha_pass_algorithm(A, B, p, O):
    arr, alpha, norm = [], [], [0]
    for i in range(0, len(A)):
        arr.append(B[i][O[0]]*p[0][i]) #  (2.8)We start off by computing the probability of having observed the first observation o1 and having been in any of the hidden states
        norm[0] += B[i][O[0]]*p[0][i]
    #print(templist)
    #Values scaled with normal factor *divide by the sum over the final α values*
    for i in range(len(A)):
        arr[i] = arr[i]/norm[0]
    alpha.append(arr)
    #we loop over all states
    for t in range(len(O)-1):
        arr = []
        norm.append(0)
        for i in range(len(A)):
            sigma = 0
            for j in range(len(A)):
                sigma += alpha[t][j]*A[j][i]
            arr.append(B[i][O[t+1]]*sigma)
            norm[t+1]+=(B[i][O[t+1]])*sigma 
        
        for i in range(len(A)):
            arr[i] = arr[i]/norm[t+1]
        alpha.append(arr)     
    return alpha, norm

#beta-pass algorithm      
def beta_pass_algorithm(A, B, O, norm):
    arr,beta = [], []
    for i in range(len(A)):
        arr.append(1 / norm[-1])
    beta.append(arr)
    observations = len(O) - 1 

    #In the following, we iterate backward
    #through the observation sequence and compute the β values
    for t in reversed(range(observations)):
        arr = []
        for i in range(len(A)):
            b = 0
            for j in range(len(A)):
                b += B[j][O[t+1]] * A[i][j]* beta[-1][j] # 2.30, compute the β values
            arr.append(b/norm[t])
        beta.append(arr)
    beta.reverse()   
    return beta

def calculate_gamma(alpha, beta, t, i, A, B, O):
    observations = len(O) - 1
    gamma = 0
    if (t != observations):
        for j in range(len(A)):
            gamma += alpha[t][i] * A[i][j] * B[j][O[t+1]] * beta[t+1][j]
        return gamma
    else:
        return alpha[t][i]
        

#Runs the overall structure of the Baum-Welch algorithm.
def Baum_Welch(A, B, p, O):
    limit, tries, prev_log_prob, N,K,T = 200, 0, float('-inf'), len(A), len(B[0]), len(O)
    alpha, norm = alpha_pass_algorithm(A, B, p, O)   #Calculate forward probabilities with the forward algorithm
    beta = beta_pass_algorithm(A, B, O, norm) #Calculate backward probabilities with the backward algorithm
 

    while(True):
        re_estimate_p(N, alpha, beta, A, B, O)
        re_estimate_A(T, N, alpha, beta, A, B, O)
        re_estimate_B(K, T, N, alpha, beta, A, B, O)
        #Calculate the log of the observations with the new estimations
        log_prob = -sum([math.log(1/a, 10) for a in norm])
        #log_prob = -log_prob
        tries += 1
        #if limit is reached or probabs converge, stop loop
        if (limit > tries and log_prob > prev_log_prob):
            prev_log_prob = log_prob
            alpha, norm = alpha_pass_algorithm(A, B, p, O)
            beta = beta_pass_algorithm(A, B, O, norm)
        else:
            return A, B

def re_estimate_p(N, alpha, beta, A, B, O):
    for i in range(N):
        p[0][i] = calculate_gamma(alpha, beta, 0, i, A, B, O)
    
def re_estimate_A(T, N, alpha, beta, A, B, O):    
    for i in range(N):
        x = 0
        for t in range(T-1):
            x += calculate_gamma(alpha, beta, t, i, A, B, O)

        for j in range(N):
            y = 0
            for t in range(T-1):
                y += alpha[t][i] * A[i][j] * B[j][O[t+1]] * beta[t+1][j]
            A[i][j] = y/x

def re_estimate_B(K, T, N, alpha, beta, A, B, O):
    for i in range(N):
            x = 0
            for t in range(T):
                x += calculate_gamma(alpha, beta, t, i, A, B, O)
            
            for j in range(K):
                y = 0
                for t in range(T):
                    if(O[t] == j):
                        y += calculate_gamma(alpha, beta, t, i, A, B, O)
                B[i][j] = y/x

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
line1 = sys.stdin.readline() #starting guess of transition matrix
line2 = sys.stdin.readline() #starting guess of emission matrix
line3 = sys.stdin.readline() #starting guess of initial state vector
line4 = sys.stdin.readline() #number of emissions (observations) and all observations
A = generate_matrix(line1)
B = generate_matrix(line2)
p = generate_matrix(line3)
O = generate_emission_list(line4)
Ax, Bx = Baum_Welch(A, B, p, O)
for i in range(len(Ax)):
    for j in range(len(Ax[0])):
        Ax[i][j] = round(Ax[i][j], 4)
for i in range(len(Bx)):
    for j in range(len(Bx[0])):
        Bx[i][j] = round(Bx[i][j], 4)       
AxP = format_output(Ax)
BxP = format_output(Bx)
#print(AxP)
#print(BxP)
out.write(AxP + '\n')
out.write(BxP)