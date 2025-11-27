# LUsolver.py
from typing import List
from .LUdecompose import LUDecomposer
import math

class LUSolver:
    def __init__(self, sig: int = 15):
        self.sig = sig
        self.decomposer = LUDecomposer(sig)

    def round_sig(self, x: float) -> float:
        if x == 0:
            return 0.0
        order = math.floor(math.log10(abs(x)))
        factor = 10 ** (order - self.sig + 1)
        return round(x / factor) * factor

    def _print_vec(self, name: str, vec: List[float], steps: List[str]):
        steps.append(f"{name}:")
        steps.append("  " + "  ".join(f"{self.round_sig(v):12.6g}" for v in vec))
        steps.append("")

    def solve(self, A: List[List[float]], b: List[float], steps: List[str]) -> List[float]:
        steps.append("\n" + "="*80)
        steps.append("            SOLVING Ax = b USING LU DECOMPOSITION")
        steps.append("="*80)

        # Step 1: Decompose
        try:
            L, U, P = self.decomposer.decompose(A.copy(), steps)
        except ValueError as e:
            steps.append(f"\nERROR: {e}")
            raise

        n = len(A)
        steps.append("Original b vector:")
        self._print_vec("b", b, steps)

        # Step 2: Apply permutation → Pb
        Pb = [b[P[i]] for i in range(n)]
        steps.append(f"Apply permutation P → Pb = P × b")
        steps.append("Permuted right-hand side:")
        self._print_vec("Pb", Pb, steps)

        # Step 3: Forward substitution Ly = Pb
        steps.append("\nFORWARD SUBSTITUTION: Solve Ly = Pb")
        steps.append("-"*60)
        y = [0.0] * n
        for i in range(n):
            y[i] = Pb[i]
            for j in range(i):
                sub = self.round_sig(L[i][j] * y[j])
                y[i] = self.round_sig(y[i] - sub)
                if abs(L[i][j]) > 1e-10:
                    steps.append(f"y[{i+1}] := Pb[{i+1}] - L[{i+1}][{j+1}]·y[{j+1}]")
                    steps.append(f"         = {Pb[i]:.6g} - {L[i][j]:.6g} × {y[j]:.6g} = {y[i]:.6g}")
            steps.append(f"→ y[{i+1}] = {y[i]:.6g}")
        self._print_vec("Solution y (Ly = Pb)", y, steps)

        # Step 4: Backward substitution Ux = y
        steps.append("\nBACKWARD SUBSTITUTION: Solve Ux = y")
        steps.append("-"*60)
        x = [0.0] * n
        for i in range(n - 1, -1, -1):
            x[i] = y[i]
            for j in range(i + 1, n):
                sub = self.round_sig(U[i][j] * x[j])
                x[i] = self.round_sig(x[i] - sub)
                if abs(U[i][j]) > 1e-10:
                    steps.append(f"x[{i+1}] := y[{i+1}] - U[{i+1}][{j+1}]·x[{j+1}]")
                    steps.append(f"         = {y[i]:.6g} - {U[i][j]:.6g} × {x[j]:.6g} = {x[i]:.6g}")
            divisor = U[i][i]
            x[i] = self.round_sig(x[i] / divisor)
            steps.append(f"x[{i+1}] := x[{i+1}] / U[{i+1}][{i+1}] = {x[i]:.6g} / {divisor:.6g} = {x[i]:.6g}")

        steps.append("\n" + "="*80)
        steps.append("                   FINAL SOLUTION")
        steps.append("="*80)
        for i, val in enumerate(x, 1):
            steps.append(f"x{i} = {self.round_sig(val):.10g}")
        steps.append("")

        return x