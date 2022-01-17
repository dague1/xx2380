import sys
import math
#kevin & david
#Given a set of observation sequenced, how do we learn the model probabilities that would generate them?
# given lambda = (A, B, pi) we can produce alpha, beta , gamma, digamma
#given alpha, beta, gamma, digamma, we can produce
#lambda_bar = (Ax, Bx, Pix)
# iterate, until Ax Bx Pix become a local optimum.
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
    Obervations = []
    num = int(floatLine[0])
    for i in range(num):
        Obervations.append(int(line[i + 1]))  
    return Obervations

def format_output(matrix):
    outputStr = ""
    outputStr += str(len(matrix)) + " " + str(len(matrix[0]))
    for i in matrix: 
        for j in i:
            outputStr +=  " " + str(j)
    return outputStr
#
#alpha: captures the probability that we are in a given state Si in time t, given all the observations before (regardless how we got there)
def alpha_pass_algorithm(A,B,pi,O):
    N = len(A)
    T = len(O)
    norm = [0 for i in range(len(O))]
    norm[0]  = 0
    alpha_list = [[0 for i in range(N)] for j in range(T)]

    for i in range(N):
        alpha_list[0][i] = pi[i]*B[i][O[0]]
        norm[0] = norm[0] + alpha_list[0][i]
    norm[0] = 1/norm[0]
    for i in range(N):
        alpha_list[0][i] = norm[0]*alpha_list[0][i]
    for t in range (1,T):
        norm[t]= 0
        for i in range(N):
            alpha_list[t][i] = 0
            for j in range(N):
                alpha_list[t][i] = alpha_list[t][i] + alpha_list[t-1][j]*A[j][i]
            alpha_list[t][i] = alpha_list[t][i]*B[i][O[t]]
            norm[t] = norm[t] + alpha_list[t][i]
        norm[t] = 1/norm[t]
        for i in range(N):
            alpha_list[t][i] = norm[t]*alpha_list[t][i]
    return alpha_list, norm
# beta: captures the probability that we are in a given state Si in time t, knowing all the observations coming after t,
# but not knowing what was observeed in the past
def beta_pass_algorithm(A,B, O,norm):
    N = len(A)
    T = len(O)
    beta_list = [[0 for i in range(N)] for j in range(T)]
    for i in range(N):
        beta_list[-1][i] = norm[-1]
    for t in range(T-2,0,-1):
        for i in range(N):
            beta_list[t][i] = 0
            for j in range(N):
                beta_list[t][i] = beta_list[t][i] + A[i][j]*B[j][O[t+1]]*beta_list[t+1][j]
            beta_list[t][i] = norm[t]*beta_list[t][i]
    return beta_list

# gamma: the probability that at time t, we are going to be in state Si, 
# knowing everything that came before, and everything coming after.
# uses alpha and beta. Multiplies both and normalizes. 

# digamma: the probability that we TRANSIT from state i to state j,
#  given everything before and after (alpha and beta) alpha_t(i) and beta_t+1(j)
#a_aib_j(O_t+1)
# No need to normalize γt(i, j) since using scaled α and β (so its alrdy prob and not likelyhood)
def compute_gammas(A, B, alpha_list, betas, O):
    N = len(A)
    T = len(O)
    di_gammas_list = [[[0 for i in range(N)] for j in range(N)] for t in range(T)]
    gammas_list = [[0 for i in range(N)] for t in range(T)]
    for t in range(T-1):
        for i in range(N):
            gammas_list[t][i] = 0
            for j in range(N):
                di_gammas_list[t][i][j] = alpha_list[t][i]*A[i][j]*B[j][O[t+1]]*betas[t+1][j] #whats the probability of EVER reaching state i?
                gammas_list[t][i] = gammas_list[t][i] + di_gammas_list[t][i][j] ##whats the probability of EVER transitioning from i to j?
    for i in range(N):
        gammas_list[-1][i] = alpha_list[-1][i]
    return gammas_list, di_gammas_list

def re_estimate_A_B_pi(A, B, O, gammas, di_gammas):
    N = len(A)
    T = len(O)
    M = len(B[0])
    pix =  [0 for i in range(N)]
    for i in range(N):
        pix[i] = gammas[0][i]
    for i in range(N):
        nam = 0
        for t in range(T-1):
            nam = nam + gammas[t][i]
        for j in range(N):
            talj = 0
            for t in range(T-1):
                talj = talj + di_gammas[t][i][j]
            A[i][j] = talj/nam #expected number of transitions from state i to j / expeted number of transitions from i
    for i in range(N):
        nam = 0
        for t in range(T):
            nam = nam + gammas[t][i]
        for j in range(M):
            talj = 0
            for t in range(T):
                if O[t] == j:
                    talj = talj + gammas[t][i]
            B[i][j] = talj/nam #expectedf number of times in state i and observing j/ expected number of times in state i
    return A, B, pix

def compute_log(norm):
    log_probability = 0
    for i in range(len(norm)):
        log_probability  = log_probability + math.log(norm[i],10)# 1000 * ((norm[i] ** (1 / 1000)) - 1)
    log_probability = -log_probability
    return log_probability
#baum welch solves for a LOCAL maximum. no way of solving for a global optimum. 
#we know that it converges to a stable good answer, but not the best. 
#an ITERATIVE, Expectation-maximization (EM) algorithm that converges to a local optimum.
def bauch(A,B,pi, O):
    alpha_list, norm = alpha_pass_algorithm(A, B, pi, O)
    beta_list = beta_pass_algorithm(A, B, O, norm)
    log_probability = compute_log(norm)
    gammas, di_gammas = compute_gammas(A, B, alpha_list, beta_list, O)
    Ax, Bx, pix = re_estimate_A_B_pi(A, B, O, gammas, di_gammas)
    return Ax,Bx,pix,log_probability

def model_loop(A, B, pi, O):
    old_log_probability = float('-inf')
    max_iterations = 30
    iterations = 0
    while(iterations<max_iterations):
        A,B,pi,log_probability = bauch(A,B,pi,O)
        iterations = iterations + 1
        if(iterations< max_iterations and log_probability>old_log_probability):
            old_log_probability = log_probability
        else:
            print(format_output(A))
            print(format_output(B))
            break

line1 = sys.stdin.readline() #starting guess of transition matrix
line2 = sys.stdin.readline() #starting guess of emission matrix
line3 = sys.stdin.readline() #starting guess of initial state vector
line4 = sys.stdin.readline() #number of emissions (Oervations) and all Oervations
A = generate_matrix(line1)
B = generate_matrix(line2)
pi = generate_matrix(line3)
pi = pi[0]
O = generate_emission_list(line4)
model_loop(A, B, pi, O)