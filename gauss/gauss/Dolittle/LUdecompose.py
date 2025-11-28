# Dolittle/LUdecompose.py
from decimal import Context

import math
from typing import List, Tuple


class LUDecomposer:
    def __init__(self, sig: int = 15):
        self.sig = sig
        self.step_strings = []
        self._current = []

    def round_sig(self, x):
        # Create a context with the desired precision
        ctx = Context(prec=self.sig)
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

    def decompose(self, A: List[List[float]]) -> Tuple[List[List[float]], List[List[float]], List[int]]:
        n = len(A)
        A = [[self.round_sig(val) for val in row] for row in A]
        P = list(range(n))

        # Step 0: Header + Initial Matrix
        self._current = [
            "=" * 80,
            "        LU DECOMPOSITION WITH PARTIAL PIVOTING (Doolittle Form)",
            "=" * 80,
            f"Precision: {self.sig} significant figures",
            "",
            "Initial Matrix A:"
        ]
        self._print_matrix("Initial Matrix A", A)
        self._flush()

        for k in range(n):
            self._current = [f"--- COLUMN {k+1} ELIMINATION ---"]

            # Find pivot
            pivot_row = max(range(k, n), key=lambda i: abs(A[i][k]))
            pivot_val = abs(A[pivot_row][k])

            if pivot_val < 1e-15:
                self._current.append("ERROR: Matrix is singular (zero pivot after rounding)")
                self._flush()
                raise ValueError("Matrix is singular")

            self._current.append(f"Searching largest pivot in column {k+1} (rows {k+1}–{n}):")
            for i in range(k, n):
                val = abs(A[i][k])
                marker = " ← BEST PIVOT" if i == pivot_row else ""
                self._current.append(f"  |A[{i+1}][{k+1}]| = {val:.6g}{marker}")

            # Swap rows
            if pivot_row != k:
                self._current.append(f"\nSWAP Row{k+1} ↔ Row{pivot_row+1}")
                A[k], A[pivot_row] = A[pivot_row], A[k]
                P[k], P[pivot_row] = P[pivot_row], P[k]
                self._print_matrix("After row swap", A)
            else:
                self._current.append("No swap needed — pivot already in place")

            pivot = A[k][k]
            self._current.append(f"Pivot = A[{k+1}][{k+1}] = {pivot:.6g}")

            # Eliminate below
            for i in range(k + 1, n):
                if abs(A[i][k]) < 1e-15:
                    continue

                multiplier = self.round_sig(A[i][k] / pivot)
                A[i][k] = multiplier  # Store in L

                self._current.append(f"\nEliminate A[{i+1}][{k+1}] → compute multiplier")
                self._current.append(f"L[{i+1}][{k+1}] = A[{i+1}][{k+1}] / pivot = {multiplier:.6g}")

                for j in range(k + 1, n):
                    old = A[i][j]
                    sub = multiplier * A[k][j]
                    A[i][j] = self.round_sig(old - sub)
                    self._current.append(
                        f"  A[{i+1}][{j+1}] := {old:.6g} − ({multiplier:.6g} × {A[k][j]:.6g}) = {A[i][j]:.6g}"
                    )

                self._print_matrix(f"After eliminating Row{i+1}", A)
                self._flush()  # One full elimination step

            self._flush()  # End of column

        # Extract L and U
        L = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
        U = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i > j:
                    L[i][j] = A[i][j]
                elif i <= j:
                    U[i][j] = A[i][j]

        # Final L, U, P
        self._current = [
            "=" * 80,
            "        FINAL L AND U MATRICES",
            "=" * 80,
            "L (Lower triangular with 1s on diagonal):",
        ]
        self._print_matrix("L", L)
        self._current.append("U (Upper triangular):")
        self._print_matrix("U", U)
        self._current.append("Permutation vector P (row order):")
        self._current.append("P = [" + ", ".join(f"Row{i+1}" for i in P) + "]")
        self._current.append("")
        self._flush()

        return L, U, P