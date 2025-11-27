# classes/lu_doolittle.py
from typing import List, Tuple, Optional
import copy

class LUDoolittle:

    def __init__(self, matrix: List[List[float]], step_by_step: bool = True):
        self.n = len(matrix)
        self.A_orig = [row[:] for row in matrix]           # Original matrix (for display)
        self.A = [row[:] for row in matrix]                # Working copy
        self.L = [[0.0 for _ in range(self.n)] for _ in range(self.n)]
        self.U = [[0.0 for _ in range(self.n)] for _ in range(self.n)]
        self.P = list(range(self.n))                       # Permutation vector: P[i] = original row now at i
        self.step_by_step = step_by_step
        self.steps = []                                    # Store messages for replay if needed

    def _log(self, msg: str):
        if self.step_by_step:
            print(msg)
        self.steps.append(msg)

    def _print_matrix(self, mat: List[List[float]], name: str):
        if not self.step_by_step:
            return
        print(f"\n{name}:")
        for i, row in enumerate(mat):
            print(f"  R{i+1}:", "  ".join(f"{x:10.4f}" for x in row))

    def decompose(self) -> Tuple[List[List[float]], List[List[float]], List[int]]:

        print("=" * 80)
        print("          LU DECOMPOSITION (DOOLITTLE + PARTIAL PIVOTING)")
        print("=" * 80)
        self._print_matrix(self.A_orig, "Original Matrix A")

        # Initialize L with identity (we'll fill lower part)
        for i in range(self.n):
            self.L[i][i] = 1.0

        for k in range(self.n):  # k = current column
            if self.step_by_step:
                print(f"\n{'─' * 20} STEP {k+1}: Processing column {k+1} {'─' * 20}")

            max_val = abs(self.A[k][k])
            max_idx = k
            for i in range(k + 1, self.n):
                if abs(self.A[i][k]) > max_val:
                    max_val = abs(self.A[i][k])
                    max_idx = i

            if max_val < 1e-12:
                raise ValueError(f"Matrix is singular (zero pivot at column {k+1})")

            if max_idx != k:
                self._log(f"\nSwap Row{k+1} ↔ Row{max_idx+1}  (pivot = {max_val:.6f})")
                self.A[k], self.A[max_idx] = self.A[max_idx], self.A[k]
                for j in range(k):
                    self.L[k][j], self.L[max_idx][j] = self.L[max_idx][j], self.L[k][j]
                self.P[k], self.P[max_idx] = self.P[max_idx], self.P[k]
                self._print_matrix(self.A, "After Row Swap")

            pivot = self.A[k][k]
            self._log(f"Pivot[{k+1},{k+1}] = {pivot:.6f}")

            for j in range(k, self.n):
                sum_u = sum(self.L[k][m] * self.U[m][j] for m in range(k))
                self.U[k][j] = self.A[k][j] - sum_u

            for i in range(k + 1, self.n):
                sum_l = sum(self.L[i][m] * self.U[m][k] for m in range(k))
                multiplier = (self.A[i][k] - sum_l) / self.U[k][k]
                self.L[i][k] = multiplier
                for j in range(k + 1, self.n):
                    self.A[i][j] -= multiplier * self.U[k][j]

            if self.step_by_step:
                self._print_matrix(self.L, f"L after step {k+1}")
                self._print_matrix(self.U, f"U after step {k+1}")

        print("\n" + "=" * 80)
        print("                    DECOMPOSITION COMPLETE")
        print("=" * 80)
        self._print_matrix(self.L, "Final L (Lower triangular with 1s on diagonal)")
        self._print_matrix(self.U, "Final U (Upper triangular)")
        print(f"Permutation vector P = {self.P}  →  (row order: { [i+1 for i in self.P] }")

        return self.L, self.U, self.P

    def solve(self, b: List[float]) -> List[float]:
        """
        Solve Ax = b using LU decomposition: Ly = Pb, Ux = y
        Returns x in ORIGINAL variable order using permutation P
        """
        if not hasattr(self, 'L'):
            raise RuntimeError("Must call decompose() first!")

        n = self.n
        # Apply permutation to b → Pb
        Pb = [b[self.P[i]] for i in range(n)]

        # Forward substitution: Ly = Pb
        y = [0.0] * n
        for i in range(n):
            sum_y = sum(self.L[i][j] * y[j] for j in range(i))
            y[i] = Pb[i] - sum_y  # since L[ii] = 1

        # Back substitution: Ux = y
        x = [0.0] * n
        for i in range(n - 1, -1, -1):
            sum_x = sum(self.U[i][j] * x[j] for j in range(i + 1, n))
            x[i] = (y[i] - sum_x) / self.U[i][i]

        return x

    def verify(self, tol: float = 1e-10):
        """Verify that P @ A = L @ U using numpy"""
        try:
            import numpy as np
        except ImportError:
            print("numpy not installed → skipping verification")
            return

        # Build correct permutation matrix P
        P_mat = np.zeros((self.n, self.n))
        for i in range(self.n):
            P_mat[i, self.P[i]] = 1.0  # P[i, original_row] = 1

        A_np = np.array(self.A_orig)
        L_np = np.array(self.L)
        U_np = np.array(self.U)

        PA = P_mat @ A_np
        LU = L_np @ U_np

        error = np.linalg.norm(PA - LU)
        print(f"\nVerification: ||PA − LU|| = {error:.2e}")
        if error < tol:
            print("LU decomposition is CORRECT!")
        else:
            print("ERROR in decomposition!")
            print("PA:")
            print(PA)
            print("LU:")
            print(LU)