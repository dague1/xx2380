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
    #print(matrix)    
    return matrix

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
    outputStr += str(len(p_A_B)) + " " + str(len(p_A_B[0]))
    for i in matrix: 
        for j in i:
            outputStr += " " + str(j)
    return outputStr

#### yay not working with classes here
## A, B and p are the input matrices
line1 = sys.stdin.readline()
line2 = sys.stdin.readline()
line3 = sys.stdin.readline()
A = generate_matrix(line1)
B = generate_matrix(line2)
p = generate_matrix(line3)
#A,B,p = get_input(sys.stdin)
p_A = matmul(p, A) 
p_A_B = matmul(p_A, B)

out.write(format_output(p_A_B))
