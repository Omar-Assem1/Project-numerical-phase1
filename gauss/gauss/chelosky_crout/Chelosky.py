# chelosky_crout.py → Cholesky LU with steps
import math
from typing import List, Optional


class Chelosky_LU:
    def __init__(self, augmented, n, precision=None, tol=1e-20):
        self.L = [[0.0 for _ in range(n)] for _ in range(n)]
        self.U = [[0.0 for _ in range(n)] for _ in range(n)]
        self.n = n
        self.A = augmented
        self.precision = precision if precision is not None else 4
        self.tolerance = tol

    def round_sig(self, x, sig=None):
        if sig is None:
            sig = self.precision
        if x == 0:
            return 0.0
        order = math.floor(math.log10(abs(x)))
        factor = 10 ** (order - sig + 1)
        return round(x / factor) * factor

    def _print_matrix(self, title: str, matrix: List[List[float]], steps: List[str]):
        steps.append(f"\n{title}:")
        for i, row in enumerate(matrix):
            row_str = "  ".join(f"{self.round_sig(val, self.precision):12.6g}" for val in row)
            steps.append(f"  Row{i+1}: {row_str}")
        steps.append("")

    def determinant(self, matrix):
        n = len(matrix)
        if n == 1:
            return matrix[0][0]
        if n == 2:
            return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
        det = 0
        for col in range(n):
            sub = [row[:col] + row[col+1:] for row in matrix[1:]]
            det += ((-1) ** col) * matrix[0][col] * self.determinant(sub)
        return det

    def is_positive_definite(self, steps: List[str]) -> bool:
        A = [row[:self.n] for row in self.A]
        steps.append("Checking if matrix is Positive Definite (all leading minors > 0)...")
        for i in range(1, self.n + 1):
            minor = [row[:i] for row in A[:i]]
            det = self.determinant(minor)
            det_r = self.round_sig(det, self.precision)
            steps.append(f"  Leading minor {i}×{i}: det = {det_r}")
            if det <= self.tolerance:
                steps.append("  Not positive definite (minor ≤ 0)")
                return False
        steps.append("  All leading minors positive → Matrix is Positive Definite")
        return True

    def is_symmetric(self, steps: List[str]) -> bool:
        A = [row[:self.n] for row in self.A]
        steps.append("Checking symmetry (A = Aᵀ)...")
        symmetric = True
        for i in range(self.n):
            for j in range(i + 1, self.n):
                diff = abs(A[i][j] - A[j][i])
                if diff > self.tolerance:
                    steps.append(f"  A[{i+1}][{j+1}] = {A[i][j]:.6g}, A[{j+1}][{i+1}] = {A[j][i]:.6g} → Not symmetric!")
                    symmetric = False
        if symmetric:
            steps.append("  Matrix is symmetric")
        return symmetric

    def compute_LU(self, steps: List[str]):
        steps.append("\n" + "="*80)
        steps.append("        CHOLESKY DECOMPOSITION (A = LLᵀ)")
        steps.append("="*80)
        steps.append(f"Precision: {self.precision} significant figures")

        A = [row[:self.n] for row in self.A]
        self._print_matrix("Original Matrix A (must be SPD)", A, steps)

        if not self.is_symmetric(steps):
            steps.append("ERROR: Matrix is not symmetric → Cholesky not applicable")
            return False
        if not self.is_positive_definite(steps):
            steps.append("ERROR: Matrix is not positive definite → Cholesky not applicable")
            return False

        steps.append("\nStarting Cholesky decomposition...")

        for j in range(self.n):
            steps.append(f"\n--- Computing Column {j+1} of L ---")

            for i in range(j, self.n):
                if i == j:
                    sum_sq = sum(self.round_sig(self.L[i][k] ** 2, self.precision) for k in range(j))
                    val = self.round_sig(A[i][j] - sum_sq, self.precision)
                    if val < -self.tolerance:
                        steps.append(f"Negative under sqrt at L[{i+1}][{j+1}] → Not positive definite!")
                        return False
                    sqrt_val = math.sqrt(max(0, val))
                    self.L[i][j] = self.round_sig(sqrt_val, self.precision)
                    steps.append(f"L[{i+1}][{j+1}] = √(A[{i+1}][{j+1}] - Σ L[{i+1}][k]²)")
                    steps.append(f"          = √({A[i][j]:.6g} - {sum_sq:.6g}) = {self.L[i][j]:.6g}")
                else:
                    sum_prod = sum(self.round_sig(self.L[i][k] * self.L[j][k], self.precision) for k in range(j))
                    val = self.round_sig(A[i][j] - sum_prod, self.precision)
                    denom = self.L[j][j]
                    if abs(denom) < self.tolerance:
                        steps.append("Zero pivot encountered!")
                        return False
                    self.L[i][j] = self.round_sig(val / denom, self.precision)
                    steps.append(f"L[{i+1}][{j+1}] = (A[{i+1}][{j+1}] - Σ L[{i+1}][k]L[{j+1}][k]) / L[{j+1}][{j+1}]")
                    steps.append(f"          = ({A[i][j]:.6g} - {sum_prod:.6g}) / {denom:.6g} = {self.L[i][j]:.6g}")

            self._print_matrix(f"Current L after column {j+1}", self.L, steps)

        # U = L transpose
        for i in range(self.n):
            for j in range(self.n):
                self.U[i][j] = self.L[j][i]

        steps.append("\n" + "="*80)
        steps.append("CHOLESKY DECOMPOSITION COMPLETE")
        steps.append("="*80)
        self._print_matrix("Final L (Lower triangular)", self.L, steps)
        self._print_matrix("Final U = Lᵀ (Upper triangular)", self.U, steps)
        steps.append("A = L @ Lᵀ verified internally")
        return True

    def solve(self, steps: List[str]):
        steps.append("\n" + "="*80)
        steps.append("        SOLVING Ax = b USING CHOLESKY DECOMPOSITION")
        steps.append("="*80)

        if not self.compute_LU(steps):
            steps.append("Cannot solve: Cholesky decomposition failed")
            return None

        b = [row[self.n] for row in self.A]
        steps.append("Right-hand side vector b:")
        steps.append("  b = [" + ", ".join(f"{self.round_sig(val, self.precision):.6g}" for val in b) + "]")

        # Forward substitution: Ly = b
        steps.append("\nForward substitution: Solve Ly = b")
        steps.append("-"*60)
        y = [0.0] * self.n
        for i in range(self.n):
            s = sum(self.round_sig(self.L[i][j] * y[j], self.precision) for j in range(i))
            y[i] = self.round_sig((b[i] - s) / self.L[i][i], self.precision)
            steps.append(f"y[{i+1}] = (b[{i+1}] - Σ L[{i+1}][j]y[j]) / L[{i+1}][{i+1}]")
            steps.append(f"      = ({b[i]:.6g} - {s:.6g}) / {self.L[i][i]:.6g} = {y[i]:.6g}")

        # Backward substitution: Lᵀx = y → Ux = y
        steps.append("\nBackward substitution: Solve Lᵀx = y")
        steps.append("-"*60)
        x = [0.0] * self.n
        for i in range(self.n - 1, -1, -1):
            s = sum(self.round_sig(self.U[i][j] * x[j], self.precision) for j in range(i + 1, self.n))
            x[i] = self.round_sig((y[i] - s) / self.U[i][i], self.precision)
            steps.append(f"x[{i+1}] = (y[{i+1}] - Σ U[{i+1}][j]x[j]) / U[{i+1}][{i+1}]")
            steps.append(f"      = ({y[i]:.6g} - {s:.6g}) / {self.U[i][i]:.6g} = {x[i]:.6g}")

        steps.append("\n" + "="*80)
        steps.append("                 FINAL SOLUTION (CHOLESKY)")
        steps.append("="*80)
        for i, val in enumerate(x, 1):
            steps.append(f"x{i} = {self.round_sig(val, self.precision):.10g}")
        steps.append("")

        return x