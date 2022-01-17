class Alpha():
    def matrix_multiplication(self,A,B):
        N = len(A)
        N_c = len(A[0])
        M_c = len(B[0])
        res = self.empty_result(A,B)
        for i in range(N):
            for j in range(N_c):
                total = 0
                for k in range(M_c):
                    total += A[i][k] * B[k][j]
                res[i][j] = total
        return res


    def empty_result(self,A,B):
        Arows = len(A)
        Bcols = len(B[0])
        res = [0]*Arows*Bcols
        empty = self.generate_matrix(Arows, Bcols, res)
        return empty

    #Initialize new Matrix
    def generate_matrix(self, rows, cols, input):
        matrix = []
        for j in range(int(rows)):
            row = []
            for i in range(int(cols)):  
                row.append(input[cols * j + i])
            matrix.append(row)
        return matrix

    def alpha_pass_algorithm_v2(self,A,B,pi,O):
        length = O[1:]
        B_t = list(map(list, zip(*B)))
        candidate_alphas = [[a * b for a, b in zip(pi, B_t[O[0]])]]
        current_alpha = candidate_alphas
        
        for concrete_observation in length:
            neo_initial_state_vector = self.matrix_multiplication(current_alpha, A)
            current_alpha = [[a * b for a, b in zip(neo_initial_state_vector[0], B_t[concrete_observation])]]
        probability = 0
        for i in range(len(current_alpha[0])):
            probability += current_alpha[0][i]
        return probability
