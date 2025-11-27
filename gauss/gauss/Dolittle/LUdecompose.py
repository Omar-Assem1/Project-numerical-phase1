# LUdecompose.py
import math
from typing import List, Tuple


class LUDecomposer:
    def __init__(self, sig: int = 15):
        self.sig = sig

    def round_sig(self, x: float) -> float:
        if x == 0:
            return 0.0
        order = math.floor(math.log10(abs(x)))
        factor = 10 ** (order - self.sig + 1)
        return round(x / factor) * factor

    def _print_matrix(self, title: str, matrix: List[List[float]], steps: List[str]):
        steps.append(f"\n{title}:")
        for i, row in enumerate(matrix):
            row_str = "  ".join(f"{self.round_sig(x):12.6g}" for x in row)
            steps.append(f"  Row{i+1}: {row_str}")
        steps.append("")

    def decompose(self, A: List[List[float]], steps: List[str]) -> Tuple[List[List[float]], List[List[float]], List[int]]:
        n = len(A)
        A = [[self.round_sig(val) for val in row] for row in A]  # working copy
        P = list(range(n))

        steps.append("\n" + "="*80)
        steps.append("        LU DECOMPOSITION WITH PARTIAL PIVOTING (Doolittle Form)")
        steps.append("="*80)
        steps.append(f"Precision: {self.sig} significant figures")
        self._print_matrix("Initial Matrix A", A, steps)

        for k in range(n):
            steps.append(f"\n--- Column {k+1} Elimination ---")

            # Find pivot
            pivot_row = max(range(k, n), key=lambda i: abs(A[i][k]))
            pivot_val = abs(A[pivot_row][k])
            if pivot_val < 1e-15:
                steps.append("Singular matrix detected (zero pivot)")
                raise ValueError("Matrix is singular")

            steps.append(f"Searching pivot in column {k+1} (rows {k+1} to {n}):")
            for i in range(k, n):
                val = abs(A[i][k])
                marker = " ← BEST PIVOT" if i == pivot_row else ""
                steps.append(f"  |A[{i+1}][{k+1}]| = {val:.6g}{marker}")

            # Swap rows
            if pivot_row != k:
                steps.append(f"\nSWAP Row{k+1} ↔ Row{pivot_row+1}")
                A[k], A[pivot_row] = A[pivot_row], A[k]
                P[k], P[pivot_row] = P[pivot_row], P[k]
                self._print_matrix("After row swap", A, steps)
            else:
                steps.append(f"No swap needed. Pivot is already in Row{k+1}")

            pivot = A[k][k]
            steps.append(f"Pivot = A[{k+1}][{k+1}] = {pivot:.6g}")

            # Elimination below
            for i in range(k + 1, n):
                if abs(A[i][k]) < 1e-15:
                    continue

                multiplier = self.round_sig(A[i][k] / pivot)
                A[i][k] = multiplier  # store in lower part (L)

                steps.append(f"\nEliminate entry below pivot → Row{i+1}")
                steps.append(f"Multiplier L[{i+1}][{k+1}] = A[{i+1}][{k+1}] / A[{k+1}][{k+1}]")
                steps.append(f"                    = {A[i][k]:.6g} / {pivot:.6g} = {multiplier:.6g}")

                for j in range(k + 1, n):
                    old = A[i][j]
                    sub = multiplier * A[k][j]
                    A[i][j] = self.round_sig(old - sub)
                    steps.append(f"  A[{i+1}][{j+1}] := {old:.6g} - ({multiplier:.6g} × {A[k][j]:.6g}) = {A[i][j]:.6g}")

                self._print_matrix(f"After eliminating Row{i+1}", A, steps)

        # Extract L and U
        L = [[0.0] * n for _ in range(n)]
        U = [[0.0] * n for _ in range(n)]

        for i in range(n):
            L[i][i] = 1.0
            for j in range(n):
                if i > j:
                    L[i][j] = A[i][j]
                elif i <= j:
                    U[i][j] = A[i][j]

        steps.append("\n" + "="*80)
        steps.append("        FINAL L (Lower) and U (Upper) Matrices")
        steps.append("="*80)
        self._print_matrix("L (unit lower triangular)", L, steps)
        self._print_matrix("U (upper triangular)", U, steps)

        steps.append("Permutation vector P (row swaps):")
        steps.append("P = " + " → ".join(f"Row{i+1}" for i in P))
        steps.append("")

        return L, U, P