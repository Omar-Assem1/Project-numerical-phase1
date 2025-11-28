# chelosky_crout.py → Crout LU
from decimal import Context
import math
from typing import List, Optional


class Crout_LU:
    def __init__(self, augmented, n, precision=4):
        self.n = n
        self.A = [row[:] for row in augmented]  # copy
        self.L = [[0.0] * n for _ in range(n)]
        self.U = [[0.0] * n for _ in range(n)]
        self.precision = precision

        # Step tracking
        self.step_strings = []
        self._current = []

    def round_sig(self, x):
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

    def compute_LU(self) -> bool:
        A = [row[:self.n] for row in self.A]

        # Header
        self._current = [
            "=" * 80,
            "           CROUT LU DECOMPOSITION (No Pivoting)",
            "=" * 80,
            f"Precision: {self.precision} significant figures",
            "",
            "Original Matrix A:"
        ]
        self._print_matrix("A", A)
        self._flush()

        # Set diagonal of U to 1
        for i in range(self.n):
            self.U[i][i] = 1.0

        for j in range(self.n):
            self._current = [f"--- PROCESSING COLUMN {j+1} ---"]

            # Compute L[i][j] for i = j..n-1
            for i in range(j, self.n):
                sum_lu = sum(self.L[i][k] * self.U[k][j] for k in range(j))
                val = self.round_sig(A[i][j] - sum_lu)
                self.L[i][j] = val

                self._current.append(f"Compute L[{i+1}][{j+1}]")
                self._current.append(f"  L[{i+1}][{j+1}] = A[{i+1}][{j+1}] − Σ(L[{i+1}][k]·U[k][{j+1}])")
                self._current.append(f"                = {A[i][j]:.6g} − {sum_lu:.6g} = {val:.6g}")

                if i == j and abs(val) < 1e-15:
                    self._current.append("ZERO PIVOT → CROUT DECOMPOSITION FAILED")
                    self._flush()
                    return False

            # Compute U[j][i] for i = j+1..n-1
            for i in range(j + 1, self.n):
                sum_lu = sum(self.L[j][k] * self.U[k][i] for k in range(j))
                val = self.round_sig((A[j][i] - sum_lu) / self.L[j][j])
                self.U[j][i] = val

                self._current.append(f"Compute U[{j+1}][{i+1}]")
                self._current.append(f"  U[{j+1}][{i+1}] = (A[{j+1}][{i+1}] − Σ(L[{j+1}][k]·U[k][{i+1}])) / L[{j+1}][{j+1}]")
                self._current.append(f"                = ({A[j][i]:.6g} − {sum_lu:.6g}) / {self.L[j][j]:.6g} = {val:.6g}")

            self._print_matrix(f"L after column {j+1}", self.L)
            self._print_matrix(f"U after column {j+1}", self.U)
            self._flush()

        # Final L and U
        self._current = [
            "=" * 80,
            "CROUT DECOMPOSITION COMPLETE",
            "=" * 80,
            "Final L (Lower triangular):",
        ]
        self._print_matrix("L", self.L)
        self._current.append("Final U (Unit upper triangular):")
        self._print_matrix("U", self.U)
        self._flush()

        return True

    def solve(self):
        b = [row[self.n] for row in self.A]

        if not self.compute_LU():
            return None

        # Forward substitution Ly = b
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

        # Backward substitution Ux = y
        self._current = [
            "BACKWARD SUBSTITUTION: Solve Ux = y",
            "-" * 60
        ]
        x = [0.0] * self.n
        for i in range(self.n - 1, -1, -1):
            s = sum(self.U[i][j] * x[j] for j in range(i + 1, self.n))
            x[i] = self.round_sig(y[i] - s)  # U[i][i] = 1
            self._current.append(
                f"x[{i+1}] = y[{i+1}] − Σ U[{i+1}][j]x[j] = {x[i]:.6g}"
            )
        self._flush()

        # Final solution
        self._current = [
            "=" * 80,
            "                 FINAL SOLUTION (CROUT LU)",
            "=" * 80
        ]
        for i, val in enumerate(x, 1):
            self._current.append(f"x{i} = {self.round_sig(val):.10g}")
        self._current.append("")
        self._flush()

        return x