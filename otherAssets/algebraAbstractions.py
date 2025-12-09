#Python algrebra exam

import numpy as np

# --- 1. Matrix Addition ---
def matrix_addition(matrix_a, matrix_b):
    """
    Calculates the sum of two matrices.
    Matrices must have the same dimensions.
    """
    try:
        result = np.array(matrix_a) + np.array(matrix_b)
        print("\n--- Matrix Addition ---")
        print("Matrix A:\n", np.array(matrix_a))
        print("Matrix B:\n", np.array(matrix_b))
        print("A + B:\n", result)
        return result
    except ValueError as e:
        print(f"\nError in Matrix Addition: {e}. Matrices must have the same dimensions.")
        return None

# --- 2. Vector Addition ---
def vector_addition(vec_u, vec_v):
    """
    Calculates the sum of two vectors.
    Vectors must have the same dimensions.
    """
    try:
        result = np.array(vec_u) + np.array(vec_v)
        print("\n--- Vector Addition ---")
        print("Vector u:", np.array(vec_u))
        print("Vector v:", np.array(vec_v))
        print("u + v:", result)
        return result
    except ValueError as e:
        print(f"\nError in Vector Addition: {e}. Vectors must have the same dimensions.")
        return None

# --- 3. Matrix Transpose ---
def matrix_transpose(matrix):
    """
    Calculates the transpose of a matrix.
    """
    transposed_matrix = np.array(matrix).T
    print("\n--- Matrix Transpose ---")
    print("Original Matrix:\n", np.array(matrix))
    print("Transposed Matrix:\n", transposed_matrix)
    return transposed_matrix

# --- 4. Determinant of a Matrix ---
def matrix_determinant(matrix):
    """
    Calculates the determinant of a square matrix.
    Handles 2x2 and 3x3 for manual verification; np.linalg.det handles any size.
    """
    matrix_np = np.array(matrix)
    if matrix_np.shape[0] != matrix_np.shape[1]:
        print("\nError in Determinant Calculation: Matrix must be square.")
        return None
    
    det = np.linalg.det(matrix_np)
    print("\n--- Matrix Determinant ---")
    print("Matrix:\n", matrix_np)
    print("Determinant:", det)
    return det

# --- 5. Solving a System of Linear Equations ---
def solve_linear_system(coefficients_matrix, constants_vector):
    """
    Solves a system of linear equations Ax = b.
    A: coefficients matrix
    b: constants vector
    """
    A = np.array(coefficients_matrix)
    b = np.array(constants_vector)

    print("\n--- Solving System of Linear Equations ---")
    print("Coefficient Matrix A:\n", A)
    print("Constants Vector b:", b)

    try:
        solution = np.linalg.solve(A, b)
        print("Solution x:", solution)
        return solution
    except np.linalg.LinAlgError as e:
        print(f"Error: {e}. The system might be singular or ill-conditioned.")
        return None

# --- 6. Matrix Multiplication ---
def matrix_multiplication(matrix_a, matrix_b):
    """
    Calculates the product of two matrices A * B.
    Number of columns in A must equal number of rows in B.
    """
    A = np.array(matrix_a)
    B = np.array(matrix_b)

    print("\n--- Matrix Multiplication ---")
    print("Matrix A:\n", A)
    print("Matrix B:\n", B)

    try:
        product = np.dot(A, B)
        print("A * B:\n", product)
        return product
    except ValueError as e:
        print(f"Error in Matrix Multiplication: {e}. Check dimensions.")
        return None

# --- 7. Vector Norm (Magnitude) ---
def vector_norm(vector):
    """
    Calculates the Euclidean norm (magnitude) of a vector.
    """
    vec_np = np.array(vector)
    norm = np.linalg.norm(vec_np)
    print("\n--- Vector Norm ---")
    print("Vector:", vec_np)
    print("Norm (Magnitude):", norm)
    return norm

# --- 8. Number of Inversions in a Permutation ---
def count_inversions(permutation):
    """
    Counts the number of inversions in a given permutation.
    An inversion (i, j) is a pair of indices such that i < j and permutation[i] > permutation[j].
    """
    inversions = 0
    n = len(permutation)
    for i in range(n):
        for j in range(i + 1, n):
            if permutation[i] > permutation[j]:
                inversions += 1
    print("\n--- Number of Inversions in a Permutation ---")
    print("Permutation:", permutation)
    print("Number of Inversions:", inversions)
    return inversions

# --- 9. Cross Product of 3D Vectors ---
def vector_cross_product(vec_u, vec_v):
    """
    Calculates the cross product of two 3D vectors.
    """
    u_np = np.array(vec_u)
    v_np = np.array(vec_v)

    if len(u_np) != 3 or len(v_np) != 3:
        print("\nError in Cross Product: Vectors must be 3-dimensional.")
        return None

    cross_prod = np.cross(u_np, v_np)
    print("\n--- Vector Cross Product ---")
    print("Vector u:", u_np)
    print("Vector v:", v_np)
    print("u x v:", cross_prod)
    return cross_prod

# --- 10. Row Echelon Form (REF) / Reduced Row Echelon Form (RREF) and Rank ---
def get_rref_and_rank(matrix):
    """
    Calculates the Reduced Row Echelon Form (RREF) of a matrix
    and determines its rank.
    Uses sympy for symbolic computation to ensure exact fractions if needed,
    but numpy.linalg.matrix_rank gives rank for float arrays.
    """
    try:
        # Use numpy for numerical RREF approximation
        # For exact RREF with fractions, symbolic libraries like sympy are better.
        # But for general concept and rank, numpy is sufficient.
        A = np.array(matrix, dtype=float)
        
        # Calculate RREF (manual implementation for clarity or use libraries like sympy)
        # For this demonstration, we'll use a common iterative approach.
        # Note: numpy doesn't have a direct RREF function.
        # Here's a simplified RREF function based on Gaussian elimination for demonstration.
        
        # Function to compute RREF (simplified for float numbers)
        def rref_numeric(matrix_in):
            A = np.array(matrix_in, dtype=float)
            rows, cols = A.shape
            lead = 0
            for r in range(rows):
                if lead >= cols:
                    break
                i = r
                while i < rows and A[i, lead] == 0:
                    i += 1
                if i != rows:
                    A[[r, i]] = A[[i, r]] # Swap rows
                    A[r] = A[r] / A[r, lead] # Scale row to make leading entry 1
                    for i in range(rows):
                        if i != r:
                            A[i] = A[i] - A[i, lead] * A[r] # Eliminate other entries in column
                    lead += 1
            return A

        rref_matrix = rref_numeric(A.copy()) # Use a copy to avoid modifying original

        # Calculate rank using numpy's built-in function
        rank = np.linalg.matrix_rank(A)

        print("\n--- Row Echelon Form (REF) / Reduced Row Echelon Form (RREF) & Rank ---")
        print("Original Matrix:\n", A)
        print("Reduced Row Echelon Form (approx.):\n", np.round(rref_matrix, decimals=5)) # Round for display
        print("Rank of the matrix:", rank)
        return rref_matrix, rank
    except Exception as e:
        print(f"\nError in RREF/Rank calculation: {e}")
        return None, None

# --- 11. Subspace Membership / Null Space Dimension ---
def check_subspace_membership_and_nullity(basis_vectors, test_vector=None):
    """
    Determines if a test_vector belongs to the subspace spanned by basis_vectors.
    Also calculates the dimension of the null space of the matrix formed by
    the basis vectors' defining equations.
    """
    print("\n--- Subspace Membership / Null Space Dimension ---")
    basis_np = np.array(basis_vectors)
    print("Basis vectors spanning subspace W:\n", basis_np)

    # To check subspace membership, we try to express the test_vector as a linear combination
    # of the basis_vectors. This involves solving a system of linear equations.
    # For a vector v to be in span{b1, b2, ...}, we need v = c1*b1 + c2*b2 + ...
    # This can be set up as a matrix equation B*c = v, where B is the matrix
    # whose columns are the basis vectors, and c is the vector of coefficients.

    # Example: If W = span{(1,0,-1), (2,1,3)}, then a vector (x,y,z) is in W if
    # (x,y,z) = c1*(1,0,-1) + c2*(2,1,3)
    # This corresponds to:
    # [1 2] [c1] = [x]
    # [0 1] [c2] = [y]
    # [-1 3]      [z]
    # This is an overdetermined system. We check consistency by checking the rank
    # of the coefficient matrix and the augmented matrix.

    # For the problem provided in the image (S = {(x,y,z,w) | x-y+z-w=0, 2x+y-z+2w=0}),
    # S is the null space of the matrix A = [[1, -1, 1, -1], [2, 1, -1, 2]].
    # The dimension of S is the nullity of A.

    # Re-implementing for the Null Space concept:
    print("\n--- Null Space / Kernel Dimension ---")
    if basis_vectors and isinstance(basis_vectors[0], list): # If basis_vectors are equations
        A_null_space = np.array(basis_vectors)
        print("Matrix A defining the null space:\n", A_null_space)
        rank_A = np.linalg.matrix_rank(A_null_space)
        num_cols = A_null_space.shape[1]
        nullity = num_cols - rank_A
        print(f"Dimension of the null space (nullity): {nullity}")
        
        if test_vector is not None:
            # Check if test_vector satisfies the equations (belongs to null space)
            test_vec_np = np.array(test_vector)
            result = np.dot(A_null_space, test_vec_np)
            is_member = np.allclose(result, 0) # Use allclose for float comparisons
            print(f"Test vector {test_vector} belongs to the null space: {is_member}")
            return nullity, is_member
        return nullity, None
    else: # If basis_vectors are literally basis vectors for a span
        print("This function primarily supports calculating null space dimension for systems of equations.")
        print("For checking vector membership in a span, you'd typically form a matrix from basis vectors as columns and check consistency.")
        return None, None


# --- Example Usage ---
if __name__ == "__main__":
    # --- Matrix Addition Example ---
    A_add = [[1, 2], [3, 4]]
    B_add = [[0, 1], [-1, 2]]
    matrix_addition(A_add, B_add)

    # --- Vector Addition Example ---
    u_vec = [2, -1, 3]
    v_vec = [1, 4, -2]
    vector_addition(u_vec, v_vec)

    # --- Matrix Transpose Example ---
    A_transpose = [[1, -1], [0, 2], [3, 4]]
    matrix_transpose(A_transpose)

    # --- Determinant of a Matrix Example ---
    A_det = [[1, 2], [3, 4]]
    matrix_determinant(A_det)
    A_det_3x3 = [[1, 2, 3], [0, 1, 4], [5, 6, 0]]
    matrix_determinant(A_det_3x3)
    # Example for determinant after row swap
    A_orig_swap = [[1,2],[3,4]]
    det_orig = np.linalg.det(A_orig_swap)
    A_swapped = [[3,4],[1,2]]
    det_swapped = np.linalg.det(A_swapped)
    print("\n--- Determinant after Row Swap ---")
    print("Original matrix:\n", A_orig_swap)
    print("Original determinant:", det_orig)
    print("Matrix after row swap:\n", A_swapped)
    print("Determinant after row swap:", det_swapped)
    print(f"Observation: det(A_swapped) = -det(A_orig_swap) is {np.isclose(det_swapped, -det_orig)}")

    # --- Solving System of Linear Equations Example ---
    # x + 2y = 5
    # 3x - y = 1
    A_sys = [[1, 2], [3, -1]]
    b_sys = [5, 1]
    solve_linear_system(A_sys, b_sys)

    # --- Matrix Multiplication Example ---
    # (1 2) * (3)
    #           (4)
    A_mult = [[1, 2]]
    B_mult = [[3], [4]]
    matrix_multiplication(A_mult, B_mult)

    # --- Vector Norm Example ---
    w_vec = [4, 0, -3]
    vector_norm(w_vec)

    # --- Number of Inversions in a Permutation Example ---
    permutation = [2, 3, 4, 1]
    count_inversions(permutation)

    # --- Cross Product of 3D Vectors Example ---
    i_vec = [1, 0, 0]
    j_vec = [0, 1, 0]
    vector_cross_product(i_vec, j_vec)
    k_vec = [0, 0, 1]
    vector_cross_product(j_vec, k_vec)

    # --- Row Echelon Form (REF) / Reduced Row Echelon Form (RREF) and Rank ---
    matrix_rref = [[1, 2, 3], [0, 1, 2]]
    get_rref_and_rank(matrix_rref)
    
    matrix_for_rref_complex = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
    get_rref_and_rank(matrix_for_rref_complex)

    # --- Subspace Membership / Null Space Dimension Example ---
    # From problem: S = {(x,y,z,w) | x-y+z-w=0, 2x+y-z+2w=0}
    # This implies A = [[1, -1, 1, -1], [2, 1, -1, 2]] and S is its null space.
    null_space_matrix = [[1, -1, 1, -1], [2, 1, -1, 2]]
    
    # Test vector (0,1,5) which should belong to the span of (1,0,-1) and (2,1,3)
    # This is checking a different problem, need to adjust based on the image's "subspace" question
    # The image asked "Cual de los siguientes vectores pertenece a W?" for W = span{(1,0,-1), (2,1,3)}
    # Let's write specific code for that problem.
    print("\n--- Subspace Membership (Span Example) ---")
    v1_span = np.array([1, 0, -1])
    v2_span = np.array([2, 1, 3])
    print(f"Subspace W is spanned by v1={v1_span} and v2={v2_span}")

    def check_span_membership(vector, basis_vectors):
        """
        Checks if a vector belongs to the span of a set of basis vectors.
        """
        # Form an augmented matrix [basis_vectors | vector] and check consistency.
        # This is equivalent to solving basis_vectors * c = vector
        matrix_A = np.column_stack(basis_vectors)
        augmented_matrix = np.column_stack((matrix_A, vector))
        
        rank_A = np.linalg.matrix_rank(matrix_A)
        rank_augmented = np.linalg.matrix_rank(augmented_matrix)
        
        if rank_A == rank_augmented:
            # If ranks are equal and rank_A == number of columns in matrix_A, it's a unique solution
            # If rank_A < number of columns, it's infinite solutions.
            # In both cases, it means the vector is in the span.
            return True
        else:
            return False

    test_vectors_span = {
        "a": [1, 1, 1],
        "b": [3, 1, 1],
        "c": [4, 2, 5],
        "d": [0, 1, 5]
    }

    for key, vec in test_vectors_span.items():
        is_in_span = check_span_membership(vec, [v1_span, v2_span])
        print(f"Vector {key} ({vec}) belongs to W: {is_in_span}")

    # For the Null Space dimension problem from the image (where S is the null space)
    print("\n--- Null Space Dimension for S from problem image ---")
    A_for_null_space = [[1, -1, 1, -1], [2, 1, -1, 2]]
    nullity_result, _ = check_subspace_membership_and_nullity(A_for_null_space)
    print(f"The dimension of subspace S (null space) is: {nullity_result}")
    
    # Test a vector for the null space membership (from the original problem, (0,1,5) is not from this problem)
    # Let's find a vector that should be in the null space if nullity is 2
    # From manual calculation: (x,y,z,w) = (-1/3*beta, alpha - 4/3*beta, alpha, beta)
    # Let alpha = 3, beta = 3: x = -1, y = 3 - 4 = -1, z = 3, w = 3. So (-1,-1,3,3) should be in null space
    test_vec_null = [-1, -1, 3, 3]
    print(f"\nChecking membership for {test_vec_null} in the null space:")
    nullity_result, is_member_null = check_subspace_membership_and_nullity(A_for_null_space, test_vec_null)