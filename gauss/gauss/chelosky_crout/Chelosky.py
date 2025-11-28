# chelosky_crout.py → Cholesky
from decimal import Context
import math
from typing import List


class Chelosky_LU:
    def __init__(self, augmented, n, precision=4):
        self.n = n
        self.A = [row[:] for row in augmented]
        self.L = [[0.0] * n for _ in range(n)]
        self.U = [[0.0] * n for _ in range(n)]
        self.precision = precision

        self.step_strings = []
        self._current = []

    def round_sig(self,x):
        # Create a context with the desired precision
        ctx = Context(prec=self.precision)
        # Normalize applies the precision to the number
        return float(ctx.create_decimal(x).normalize())

    def _flush(self):
        if self._current:
            self.step_strings.append("\n".join(self._current))
            self._current = []

    def _print_matrix(self, title: str, matrix: List[List[float]]):
        self._current.append(f"\n{title}:")
        for i, row in enumerate(matrix):
            row_str = "  ".join(f"{self.round_sig(x):12.6g}" for x in row)
            self._current.append(f"  Row{i+1}: {row_str}")
        self._current.append("")

    def _is_symmetric_and_pd(self) -> bool:
        A = [row[:self.n] for row in self.A]
        tol = 1e-12

        # Check symmetry
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if abs(A[i][j] - A[j][i]) > tol:
                    self._current.append("ERROR: Matrix is not symmetric!")
                    self._flush()
                    return False

        # Check positive definite via Cholesky (we'll detect negative sqrt)
        return True  # We'll catch issues during decomposition

    def compute_LU(self) -> bool:
        A = [row[:self.n] for row in self.A]

        self._current = [
            "=" * 80,
            "        CHOLESKY DECOMPOSITION (A = LLᵀ)",
            "=" * 80,
            f"Precision: {self.precision} significant figures",
            "",
            "Original Matrix A (must be Symmetric Positive Definite):"
        ]
        self._print_matrix("A", A)

        if not self._is_symmetric_and_pd():
            return False

        self._current.append("Matrix is symmetric → proceeding with Cholesky")
        self._flush()

        for j in range(self.n):
            self._current = [f"--- COMPUTING COLUMN {j+1} OF L ---"]

            for i in range(j, self.n):
                if i == j:
                    # Diagonal: L[ii] = sqrt(A[ii] - sum L[ik]^2)
                    sum_sq = sum(self.L[i][k] ** 2 for k in range(j))
                    val = A[i][j] - sum_sq
                    if val < -1e-10:
                        self._current.append("NEGATIVE VALUE UNDER SQUARE ROOT → NOT POSITIVE DEFINITE!")
                        self._flush()
                        return False
                    sqrt_val = math.sqrt(max(0.0, val))
                    self.L[i][j] = self.round_sig(sqrt_val)
                    self._current.append(
                        f"L[{i+1}][{j+1}] = √(A[{i+1}][{j+1}] − Σ L[{i+1}][k]²) = √({A[i][j]:.6g} − {sum_sq:.6g}) = {self.L[i][j]:.6g}"
                    )
                else:
                    # Off-diagonal
                    sum_prod = sum(self.L[i][k] * self.L[j][k] for k in range(j))
                    val = (A[i][j] - sum_prod) / self.L[j][j]
                    self.L[i][j] = self.round_sig(val)
                    self._current.append(
                        f"L[{i+1}][{j+1}] = (A[{i+1}][{j+1}] − Σ L[{i+1}][k]L[{j+1}][k]) / L[{j+1}][{j+1}]"
                    )
                    self._current.append(
                        f"                = ({A[i][j]:.6g} − {sum_prod:.6g}) / {self.L[j][j]:.6g} = {self.L[i][j]:.6g}"
                    )

            self._print_matrix(f"L after column {j+1}", self.L)
            self._flush()

        # Set U = L^T
        for i in range(self.n):
            for j in range(self.n):
                self.U[i][j] = self.L[j][i]

        self._current = [
            "=" * 80,
            "CHOLESKY DECOMPOSITION COMPLETE",
            "=" * 80,
            "Final L (Lower triangular):",
        ]
        self._print_matrix("L", self.L)
        self._current.append("Final U = Lᵀ (Upper triangular):")
        self._print_matrix("U", self.U)
        self._flush()

        return True

    def solve(self):
        b = [row[self.n] for row in self.A]

        if not self.compute_LU():
            return None

        # Forward: Ly = b
        self._current = [
            "FORWARD SUBSTITUTION: Solve Ly = b",
            "-" * 60
        ]
        y = [0.0] * self.n
        for i in range(self.n):
            s = sum(self.L[i][j] * y[j] for j in range(i))
            y[i] = self.round_sig((b[i] - s) / self.L[i][i])
            self._current.append(
                f"y[{i+1}] = (b[{i+1}] − Σ L[{i+1}][j]y[j]) / L[{i+1}][{i+1}] = {y[i]:.6g}"
            )
        self._flush()

        # Backward: Lᵀx = y
        self._current = [
            "BACKWARD SUBSTITUTION: Solve Lᵀx = y",
            "-" * 60
        ]
        x = [0.0] * self.n
        for i in range(self.n - 1, -1, -1):
            s = sum(self.U[i][j] * x[j] for j in range(i + 1, self.n))
            x[i] = self.round_sig((y[i] - s) / self.U[i][i])
            self._current.append(
                f"x[{i+1}] = (y[{i+1}] − Σ Lᵀ[{i+1}][j]x[j]) / Lᵀ[{i+1}][{i+1}] = {x[i]:.6g}"
            )
        self._flush()

        # Final
        self._current = [
            "=" * 80,
            "                 FINAL SOLUTION (CHOLESKY)",
            "=" * 80
        ]
        for i, val in enumerate(x, 1):
            self._current.append(f"x{i} = {self.round_sig(val):.10g}")
        self._current.append("")
        self._flush()

        return x