# chelosky_crout.py → Crout LU with steps
import math
from typing import List, Tuple
class Crout_LU:
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

    def compute_LU(self, steps: List[str]) -> bool:
        steps.append("\n" + "="*80)
        steps.append("           CROUT LU DECOMPOSITION (No Pivoting)")
        steps.append("="*80)
        steps.append(f"Precision: {self.precision} significant figures")

        A = [row[:self.n] for row in self.A]
        self._print_matrix("Original Matrix A", A, steps)

        for i in range(self.n):
            self.U[i][i] = 1.0

        for j in range(self.n):
            steps.append(f"\n--- Processing Column {j+1} ---")

            # Compute L[i][j] for i = j to n-1
            for i in range(j, self.n):
                sum_lu = sum(self.round_sig(self.L[i][k] * self.U[k][j], self.precision) for k in range(j))
                val = self.round_sig(A[i][j] - sum_lu, self.precision)
                self.L[i][j] = val
                steps.append(f"L[{i+1}][{j+1}] = A[{i+1}][{j+1}] - Σ(L[{i+1}][k]U[k][{j+1}])")
                steps.append(f"          = {A[i][j]:.6g} - {sum_lu:.6g} = {val:.6g}")

            # Compute U[j][i] for i = j+1 to n-1
            for i in range(j + 1, self.n):
                if abs(self.L[j][j]) < self.tolerance:
                    steps.append(f"Zero pivot at L[{j+1}][{j+1}] → Crout fails!")
                    return False
                sum_lu = sum(self.round_sig(self.L[j][k] * self.U[k][i], self.precision) for k in range(j))
                val = self.round_sig((A[j][i] - sum_lu) / self.L[j][j], self.precision)
                self.U[j][i] = val
                steps.append(f"U[{j+1}][{i+1}] = (A[{j+1}][{i+1}] - Σ(L[{j+1}][k]U[k][{i+1}])) / L[{j+1}][{j+1}]")
                steps.append(f"          = ({A[j][i]:.6g} - {sum_lu:.6g}) / {self.L[j][j]:.6g} = {val:.6g}")

            self._print_matrix(f"Current L (after col {j+1})", self.L, steps)
            self._print_matrix(f"Current U (after col {j+1})", self.U, steps)

        steps.append("\n" + "="*80)
        steps.append("CROUT DECOMPOSITION COMPLETE")
        steps.append("="*80)
        self._print_matrix("Final L (Lower triangular)", self.L, steps)
        self._print_matrix("Final U (Unit upper triangular)", self.U, steps)
        return True

    def solve(self, steps: List[str]):
        steps.append("\n" + "="*80)
        steps.append("        SOLVING Ax = b USING CROUT LU")
        steps.append("="*80)

        if not self.compute_LU(steps):
            steps.append("Cannot solve: Crout decomposition failed")
            return None

        b = [row[self.n] for row in self.A]
        steps.append("Right-hand side b:")
        steps.append("  b = [" + ", ".join(f"{self.round_sig(val, self.precision):.6g}" for val in b) + "]")

        # Forward substitution: Ly = b
        steps.append("\nForward substitution: Ly = b")
        y = [0.0] * self.n
        for i in range(self.n):
            s = sum(self.round_sig(self.L[i][j] * y[j], self.precision) for j in range(i))
            y[i] = self.round_sig((b[i] - s) / self.L[i][i], self.precision)
            steps.append(f"y[{i+1}] = (b[{i+1}] - Σ L[{i+1}][j]y[j]) / L[{i+1}][{i+1}] = {y[i]:.6g}")

        # Backward substitution: Ux = y
        steps.append("\nBackward substitution: Ux = y")
        x = [0.0] * self.n
        for i in range(self.n - 1, -1, -1):
            s = sum(self.round_sig(self.U[i][j] * x[j], self.precision) for j in range(i + 1, self.n))
            x[i] = self.round_sig(y[i] - s, self.precision)  # U[ii] = 1
            steps.append(f"x[{i+1}] = y[{i+1}] - Σ U[{i+1}][j]x[j] = {y[i]:.6g} - {s:.6g} = {x[i]:.6g}")

        steps.append("\n" + "="*80)
        steps.append("                 FINAL SOLUTION (CROUT LU)")
        steps.append("="*80)
        for i, val in enumerate(x, 1):
            steps.append(f"x{i} = {self.round_sig(val, self.precision):.10g}")
        steps.append("")

        return x