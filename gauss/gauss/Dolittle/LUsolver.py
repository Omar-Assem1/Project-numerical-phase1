# Dolittle/LUsolver.py
import math
from typing import List
from .LUdecompose import LUDecomposer
from decimal import Context


class LUSolver:
    def __init__(self, sig: int = 15):
        self.sig = sig
        self.decomposer = LUDecomposer(sig)
        self.step_strings = []
        self._current = []

    def round_sig(self, x):
        # Create a context with the desired precision
        ctx = Context(prec= self.sig)
        # Normalize applies the precision to the number
        return float(ctx.create_decimal(x).normalize())

    def _flush(self):
        if self._current:
            self.step_strings.append("\n".join(self._current))
            self._current = []

    def _print_vec(self, name: str, vec: List[float]):
        self._current.append(f"{name}:")
        self._current.append("  " + "  ".join(f"{self.round_sig(v):12.6g}" for v in vec))
        self._current.append("")

    def solve(self, A: List[List[float]], b: List[float]) -> List[float]:
        self.step_strings = []  # Reset

        # Step 1: Decompose
        self._current = [
            "=" * 80,
            "            SOLVING Ax = b USING LU DECOMPOSITION",
            "=" * 80,
            ""
        ]
        self._flush()

        try:
            L, U, P = self.decomposer.decompose(A.copy())
        except ValueError as e:
            self._current = [f"ERROR: {e}"]
            self._flush()
            raise

        # Add decomposer steps
        self.step_strings.extend(self.decomposer.step_strings)

        n = len(A)

        # Step 2: Permute b → Pb
        Pb = [b[P[i]] for i in range(n)]
        self._current = [
            "APPLYING PERMUTATION → Pb = P × b",
            "Original b:",
        ]
        self._print_vec("b", b)
        self._current.append("After permutation (Pb):")
        self._print_vec("Pb", Pb)
        self._flush()

        # Step 3: Forward substitution Ly = Pb
        self._current = [
            "FORWARD SUBSTITUTION: Solve Ly = Pb",
            "-" * 60
        ]
        y = [0.0] * n
        for i in range(n):
            y[i] = Pb[i]
            for j in range(i):
                if abs(L[i][j]) > 1e-10:
                    sub = self.round_sig(L[i][j] * y[j])
                    y[i] = self.round_sig(y[i] - sub)
                    self._current.append(
                        f"y[{i+1}] := Pb[{i+1}] − L[{i+1}][{j+1}]·y[{j+1}] = {y[i]:.6g}"
                    )
            self._current.append(f"→ y[{i+1}] = {self.round_sig(y[i]):.6g}")
        self._print_vec("Solution y", y)
        self._flush()

        # Step 4: Back substitution Ux = y
        self._current = [
            "BACKWARD SUBSTITUTION: Solve Ux = y",
            "-" * 60
        ]
        x = [0.0] * n
        for i in range(n - 1, -1, -1):
            x[i] = y[i]
            for j in range(i + 1, n):
                if abs(U[i][j]) > 1e-10:
                    sub = self.round_sig(U[i][j] * x[j])
                    x[i] = self.round_sig(x[i] - sub)
                    self._current.append(
                        f"x[{i+1}] := y[{i+1}] − U[{i+1}][{j+1}]·x[{j+1}] = {x[i]:.6g}"
                    )
            divisor = U[i][i]
            x[i] = self.round_sig(x[i] / divisor)
            self._current.append(
                f"x[{i+1}] := {x[i]:.6g} / {divisor:.6g} = {self.round_sig(x[i]):.6g}"
            )
        self._flush()

        # Final solution
        self._current = [
            "=" * 80,
            "                   FINAL SOLUTION",
            "=" * 80
        ]
        for i, val in enumerate(x, 1):
            self._current.append(f"x{i} = {self.round_sig(val):.10g}")
        self._current.append("")
        self._flush()

        return x