# classes_for_gauss_jordan/rref_solver.py
import math


class RREFSolver:
    def __init__(self, rref_matrix, rank, n_vars, pivot_positions, precision=6):
        self.M = rref_matrix
        self.rank = rank
        self.n = n_vars
        self.pivot_positions = pivot_positions
        self.pivot_cols = [col for row, col in pivot_positions]
        self.free_cols = [j for j in range(n_vars) if j not in self.pivot_cols]
        self.precision = precision

    def _fmt(self, x):
        if abs(x) < 1e-10:
            return "0"
        if abs(x) < 10 and abs(x) >= 1e-4:
            return f"{x:.{self.precision}f}"
        else:
            return f"{x:.{self.precision}g}"

    def solve(self, steps):
        steps.append("\n" + "="*80)
        steps.append("                        SOLUTION INTERPRETATION")
        steps.append("="*80)

        # Check inconsistency
        for i in range(self.rank, self.n):
            if abs(self.M[i][self.n]) > 1e-8:
                steps.append("Inconsistent system! → NO SOLUTION")
                steps.append(f"Row {i+1} reads: 0 = {self.M[i][self.n]:.{self.precision}g}")
                steps.append("")
                return

        num_free = len(self.free_cols)

        if num_free == 0:
            steps.append("Unique solution found!")
            steps.append("-" * 50)
            for row_idx, col in self.pivot_positions:
                val = self.M[row_idx][self.n]
                steps.append(f"  x{col+1} = {self._fmt(val)}")
            steps.append("-" * 50)
        else:
            steps.append(f"Infinite solutions! → {num_free} free variable(s)")
            steps.append(f"Free variables: {', '.join(f'x{j+1}' for j in self.free_cols)}")
            steps.append("\nParametric solution:")
            steps.append("-" * 60)

            particular = [0.0] * self.n
            coefficients = [[0.0 for _ in range(num_free)] for _ in range(self.n)]

            for r, c in self.pivot_positions:
                particular[c] = self.M[r][self.n]
                for idx, free_col in enumerate(self.free_cols):
                    coefficients[c][idx] = -self.M[r][free_col]

            for var in range(self.n):
                if var in self.pivot_cols:
                    terms = [self._fmt(particular[var])]
                    for idx, free_col in enumerate(self.free_cols):
                        coeff = coefficients[var][idx]
                        if abs(coeff) > 1e-10:
                            sign = " - " if coeff > 0 else " + "
                            val = abs(coeff)
                            terms.append(f"{sign}{self._fmt(val)} t{idx+1}")
                    expr = "".join(terms).strip()
                    if expr.startswith("+"):
                        expr = expr[1:].strip()
                    if expr.startswith("-"):
                        expr = "- " + expr[2:].strip()
                    steps.append(f"  x{var+1} = {expr}")
                else:
                    idx = self.free_cols.index(var)
                    steps.append(f"  x{var+1} = t{idx+1}  (free)")

            steps.append("-" * 60)
            if num_free == 1:
                steps.append("Let t₁ ∈ ℝ")
            else:
                steps.append(f"Let t₁, t₂, ..., t_{num_free} ∈ ℝ")
        steps.append("")