import math
class HMM3():


    def __init__(self):
        self.xa = []
    #non exact ln function but quite fast (from stackoverflow)
    def ln(self,x):
        n = 1000.0
        return n * ((x ** (1 / n)) - 1)

    def forward_algorithm(self,A,B,pi,Obs):
        c = [None for i in range(len(Obs))]
        c[0]  = 0
        alphas = [[None for i in range(len(A))] for j in range(len(Obs))]

        for i in range(len(A)):
            alphas[0][i] = pi[i]*B[i][Obs[0]]
            c[0] = c[0] + alphas[0][i]
        c[0] = 1/c[0]
        for i in range(len(A)):
            alphas[0][i] = c[0]*alphas[0][i]
        for t in range (1,len(Obs)):
            c[t]= 0
            for i in range(len(A)):
                alphas[t][i] = 0
                for j in range(len(A)):
                    alphas[t][i] = alphas[t][i] + alphas[t-1][j]*A[j][i]
                alphas[t][i] = alphas[t][i]*B[i][Obs[t]]

                c[t] = c[t] + alphas[t][i]
            c[t] = 1/c[t]
            for i in range(len(A)):
                alphas[t][i] = c[t]*alphas[t][i]
        return alphas, c

    def backward_algorithm(self,A,B,pi,Obs, c):
        betas = [[None for i in range(len(A))] for j in range(len(Obs))]
        for i in range(len(A)):
            betas[-1][i] = c[-1]
        for t in range(len(Obs)-2,0,-1):
            for i in range(len(A)):
                betas[t][i] = 0
                for j in range(len(A)):
                    betas[t][i] = betas[t][i] + A[i][j]*B[j][Obs[t+1]]*betas[t+1][j]
                betas[t][i] = c[t]*betas[t][i]
        return betas

    def gammas(self, A, B, alphas, betas, Obs):
        di_gammas = [[[None for i in range(len(A))] for j in range(len(A))] for t in range(len(Obs))]
        gammas = [[None for i in range(len(A))] for t in range(len(Obs))]
        for t in range(len(Obs)-1):
            for i in range(len(A)):
                gammas[t][i] = 0
                for j in range(len(A)):
                    di_gammas[t][i][j] = alphas[t][i]*A[i][j]*B[j][Obs[t+1]]*betas[t+1][j]
                    gammas[t][i] = gammas[t][i] + di_gammas[t][i][j]
        for i in range(len(A)):
            gammas[-1][i] = alphas[-1][i]
        return gammas, di_gammas


    def restimate_pi(self,gamma,A):
        new_pi =  [None for i in range(len(A))]
        for i in range(len(A)):
            new_pi[i] = gamma[0][i]
        return new_pi

    def restimate_A(self, A,Obs,gammas, digammas):

        for i in range(len(A)):
            denom = 0
            for t in range(len(Obs)-1):
                denom = denom + gammas[t][i]
            for j in range(len(A)):
                numer = 0
                for t in range(len(Obs)-1):
                    numer = numer + digammas[t][i][j]
                A[i][j] = numer/denom
        return A

    def restimate_B(self,A, B, Obs, gammas):
        for i in range(len(A)):
            denom = 0
            for t in range(len(Obs)):
                denom = denom + gammas[t][i]
            for j in range(len(B[0])):
                numer = 0
                for t in range(len(Obs)):
                    if Obs[t] == j:
                        numer = numer + gammas[t][i]
                B[i][j] = numer/denom
        return B


    def compute_log(self, c):
        logProb = 0
        for i in range(len(c)):
            logProb  = logProb + self.ln(c[i])
        logProb = -logProb
        return logProb

    def algorithm(self,A,B,pi, Obs):
        alphas, c = self.forward_algorithm(A, B, pi, Obs)
        betas = self.backward_algorithm(A, B, pi, Obs, c)
        gammas, digammas = self.gammas(A, B, alphas, betas, Obs)
        new_pi = self.restimate_pi(gammas,A)
        new_A = self.restimate_A(A, Obs, gammas, digammas)
        new_B = self.restimate_B(A, B, Obs, gammas)
        logProb = self.compute_log(c)
        return new_A,new_B,new_pi,logProb


    def create_output(self, A, B):
        outputA = []
        outputA.append(str(len(A)))
        outputA.append(str(len(A[0])))
        for i in range(len(A)):
            for j in range(len(A)):
                outputA.append(str(A[i][j]))
        seperator = " "
        h = seperator.join(map(str, outputA))
        print(h)
        outputB = []
        outputB.append(str(len(B)))
        outputB.append(str(len(B[0])))
        for i in range(len(B)):
            for j in range(len(B[0])):
                outputB.append(str(B[i][j]))
        seperator = " "
        bsep = seperator.join(map(str, outputB))
        print(bsep)

    def run_learningalgorithm(self, A, B, pi, Obs):
        maxIters = 30
        iters = 0
        oldLogProb = float('-inf')
        while (iters < maxIters):
            A, B, pi, logProb = self.algorithm(A, B, pi, Obs)
            iters = iters + 1
            if (iters < maxIters and logProb > oldLogProb):
                oldLogProb = logProb
            else:
                return A,B,pi
