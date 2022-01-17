class Bauch():
    def alpha_pass_algorithm(self,A,B,pi,O):
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

    def beta_pass_algorithm(self, A, B, O, norm):
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
    def compute_gammas(self,A, B, alpha_list, betas, O):
        N = len(A)
        T = len(O)
        di_gammas_list = [[[0 for i in range(N)] for j in range(N)] for t in range(T)]
        gammas_list = [[0 for i in range(N)] for t in range(T)]
        for t in range(T-1):
            for i in range(N):
                gammas_list[t][i] = 0
                for j in range(N):
                    di_gammas_list[t][i][j] = alpha_list[t][i]*A[i][j]*B[j][O[t+1]]*betas[t+1][j]
                    gammas_list[t][i] = gammas_list[t][i] + di_gammas_list[t][i][j]
        for i in range(N):
            gammas_list[-1][i] = alpha_list[-1][i]
        return gammas_list, di_gammas_list

    def re_estimate_A_B_pi(self, A, B, O, gammas, di_gammas):
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
                A[i][j] = talj/nam
        for i in range(N):
            nam = 0
            for t in range(T):
                nam = nam + gammas[t][i]
            for j in range(M):
                talj = 0
                for t in range(T):
                    if O[t] == j:
                        talj = talj + gammas[t][i]
                B[i][j] = talj/nam
        return A, B, pix

    def compute_log(self, norm):
        log_probability = 0
        for i in range(len(norm)):
            log_probability  = log_probability + 1000 * ((norm[i] ** (1 / 1000)) - 1)
        log_probability = -log_probability
        return log_probability

    def bauch(self, A,B,pi, O):
        alpha_list, norm = self.alpha_pass_algorithm(A, B, pi, O)
        beta_list = self.beta_pass_algorithm(A, B, O, norm)
        log_probability = self.compute_log(norm)
        gammas, di_gammas = self.compute_gammas(A, B, alpha_list, beta_list, O)
        Ax, Bx, pix = self.re_estimate_A_B_pi(A, B, O, gammas, di_gammas)
        return Ax,Bx,pix,log_probability
    def model_loop(self,A, B, pi, O):
        old_log_probability = float('-inf')
        max_iterations = 30
        iterations = 0
        while(iterations<max_iterations):
            A,B,pi,log_probability = self.bauch(A,B,pi,O)
            iterations = iterations + 1
            if(max_iterations > iterations and old_log_probability < log_probability):
                old_log_probability = log_probability
            else:
                return A,B, pi
  