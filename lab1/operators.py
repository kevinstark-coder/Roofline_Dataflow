from operator_base import Operator


class RELU(Operator):
    # [B, L] -> RELU -> [B, L]
    def __init__(self, dim):
        super().__init__(dim=dim)
        self.op_type = 'RELU'

    def get_effective_dim_len(self):
        return 2

    ################################################################
    ## TODO 1.A
    ################################################################
    def get_tensors(self):
        # Function Objective = Derive the number of input & output elements in the Relu operation.
        # Use 2 dimensions of Relu inputs to derive the values.
        # Function  Output = Number of elements of input_a  and output
        B, L = self.dim[:self.get_effective_dim_len()]
        input_a = B*L
        input_b = 0 # Since single input operation, we will keep this 0.
        output = B*L
        return input_a, input_b, output

    ################################################################
    ## TODO 1.A
    ################################################################      
    def get_num_ops(self):
        # Function Objective = Derive the number of  individual operations in Relu.
        # Relu(x) = x if x>0 else 0.
        # Consider relu on each element as 1 operation.
        # Use 2 dimensions of Relu inputs to derive the values.
        # Function Output = Number of elements of input_a  and output
        B, L = self.dim[:self.get_effective_dim_len()]
        num_ops = B*L
        return num_ops
    
class ADD(Operator):
    # [B, X, Y] + [B, X, Y] = [B, X, Y]

    def __init__(self, dim):
        super().__init__(dim=dim)
        self.op_type = 'ADD'

    def get_effective_dim_len(self):
        return 3

    ################################################################
    ## TODO 1.B
    ################################################################
    def get_tensors(self):
        # Function Objective = Derive the number of input & output elements in the addition of 2 tensors.
        # Use dimensions of the tensor:  batch , tensor height and tensor width to derive the values.
        # Function Output = Number of elements of input_a , input_b and output
        B, X, Y = self.dim[:self.get_effective_dim_len()]
        input_a = B* X* Y
        input_b = B* X* Y
        output = B* X* Y
        return input_a, input_b, output

    ################################################################
    ## TODO 1.B
    ################################################################      
    def get_num_ops(self):
        ## Function get_num_ops for 2D vector addition.
        # Addition of each element is consider a single operation, i.e. 8+8+0+3 = 3 operations.
        # Function Objective = Derive the number of operations (Add)
        # Use dimensions of the tensor:  batch , tensor height and tensor width to derive the values.
        # Function  Output = Total number of individual operation in the given 2D convolution.
        B, X, Y = self.dim[:self.get_effective_dim_len()]
        num_ops = B* X* Y
        return num_ops
 
class GEMM(Operator):
    # [B, M, K] * [K, N] = [B, M, N]
    def __init__(self, dim):
        super().__init__(dim=dim)
        self.op_type = 'GEMM'

    def get_effective_dim_len(self):
        return 4


    ################################################################
    ## TODO 1.C
    ################################################################     
    def get_tensors(self):
        # Function Objective = Derive the number of input & output elements in the multiplication of 2 matrix.
        # Use dimensions of the matmul to derive the values.
        # A x B = C; Dim (A) = [B , M, K], Dim (B) = [K, N], Dim (C) = [B, M, N], Reduction dimension = K
        # Function Output = Number of elements of input_a , input_b and output
        B, M, N, K = self.dim[:self.get_effective_dim_len()]
        input_a = B * M * K
        input_b = K * M
        output = B * M * N
        return input_a, input_b, output


    ################################################################
    ## TODO 1.C
    ################################################################     
    def get_num_ops(self):
        # Function Objective = Derive the number of operations (multiplication + additions)
        # Use dimensions of the matmul to derive the values.
        # A x B = C; Dim (A) = [B , M, K], Dim (B) = [K, N], Dim (C) = [B, M, N], Reduction dimension = K
        # Function Output = Total number of individual operation in the given Matmul.
        B, M, N, K = self.dim[:self.get_effective_dim_len()]
        # Addition of each element is consider a single operation, i.e. 8+8+0+3 = 3 operations.
        # Multiplication of each element is consider a single operation, i.e. 2x4 = 1 operation.
        # Hint : You can round the number of operations to the nearest. 
        # Ex: Matrix of 1x100 and 100x1 will require 100 multiplications and 99 additions ~ round this to 100 + 100 = 200
        num_ops = B * M * N * 2 * K * K
        return num_ops
    
    
class CONV2D(Operator):
    # [B, C, X, Y] @ [K, C, R, S] = [B, K, X, Y]
    def __init__(self, dim):
        super().__init__(dim=dim)
        self.op_type = 'CONV2D'

    def get_effective_dim_len(self):
        return 7

    ################################################################
    ## TODO 1.D
    ################################################################
    def get_tensors(self):
        ## Function get_tensors for 2D Convolution.
        # Function Objective = Derive the number of elements of inputs and outputs in the convolution.
        # Use dimensions of the Conv2D to derive the values.
        # Conv2D(A, B) = C; Dim (A) = [B, C, X, Y], Dim (B) = [K, C, R, S], Dim (C) = [B, K, X, Y].
        # Function  Output = Number of elements of input_a , input_b and output
        B, K, C, X, Y, R, S = self.dim[:self.get_effective_dim_len()]
        input_a = B * C * X * Y
        input_b = K * R * S * C
        output = B * S * Y * K
        return input_a, input_b, output

    ################################################################
    ## TODO 1.D
    ################################################################      
    def get_num_ops(self):
        ## Function get_num_ops for 2D Convolution
        # Function Objective = Derive the number of operations (Add, multiplication, etc)
        # Use dimensions of the Conv2D to derive the values. Assume Stride = 1
        # Conv2D(A, B) = C; Dim (A) = [B, C, X, Y], Dim (B) = [K, C, R, S], Dim (C) = [B, K, X, Y].
        # Function  Output = Total number of individual operation in the given 2D convolution.
        B, K, C, X, Y, R, S = self.dim[:self.get_effective_dim_len()]
        num_ops = B * K * 2 * R * S * C * X * Y
        return num_ops