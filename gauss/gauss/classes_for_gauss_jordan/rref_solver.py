# classes_for_gauss_jordan/rref_solver.py
class RREFSolver:
    def __init__(self, rref_matrix, rank, n_vars, pivot_positions, precision=6):
        self.M = rref_matrix
        self.rank = rank
        self.n = n_vars
        self.pivot_positions = pivot_positions
        self.pivot_cols = [col for _, col in pivot_positions]
        self.free_cols = [j for j in range(n_vars) if j not in self.pivot_cols]
        self.precision = precision

        self.step_strings = []
        self._current = []

    def _fmt(self, x):
        if abs(x) < 1e-10:
            return "0"
        return f"{x:.{self.precision}g}"

    def _flush(self):
        if self._current:
            self.step_strings.append("\n".join(self._current))
            self._current = []

    def solve(self):
        self._current = [
            "=" * 80,
            "                        SOLUTION INTERPRETATION",
            "=" * 80,
            ""
        ]

        # Inconsistency
        for i in range(self.rank, self.n):
            if abs(self.M[i][self.n]) > 1e-8:
                self._current.extend([
                    "INCONSISTENT SYSTEM → NO SOLUTION",
                    f"Row {i+1} implies: 0 = {self._fmt(self.M[i][self.n])}",
                    ""
                ])
                self._flush()
                return None

        num_free = len(self.free_cols)

        if num_free == 0:
            self._current.append("UNIQUE SOLUTION")
            self._current.append("-" * 50)
            solution = []
            for row_idx, col in self.pivot_positions:
                val = self.M[row_idx][self.n]
                solution.append(val)
                self._current.append(f"  x{col+1} = {self._fmt(val)}")
            self._current.append("-" * 50)
            self._flush()
            return solution
        else:
            self._current.append(f"INFINITE SOLUTIONS → {num_free} free variable(s)")
            self._current.append(f"Free variables: {', '.join(f'x{j+1}' for j in self.free_cols)}")
            self._current.append("")
            self._current.append("PARAMETRIC SOLUTION:")
            self._current.append("-" * 60)

            particular = [0.0] * self.n
            coeffs = [[0.0] * num_free for _ in range(self.n)]

            for r, c in self.pivot_positions:
                particular[c] = self.M[r][self.n]
                for idx, fcol in enumerate(self.free_cols):
                    coeffs[c][idx] = -self.M[r][fcol]

            for var in range(self.n):
                if var in self.pivot_cols:
                    terms = [self._fmt(particular[var])]
                    for idx, fcol in enumerate(self.free_cols):
                        coef = coeffs[var][idx]
                        if abs(coef) > 1e-10:
                            sign = " - " if coef > 0 else " + "
                            terms.append(f"{sign}{self._fmt(abs(coef))} t{idx+1}")
                    expr = "".join(terms).replace("+ -", "- ").replace("  ", " ").strip()
                    if expr.startswith("- "):
                        expr = "- " + expr[2:]
                    self._current.append(f"  x{var+1} = {expr}")
                else:
                    idx = self.free_cols.index(var)
                    self._current.append(f"  x{var+1} = t{idx+1}  (free)")

            self._current.append("-" * 60)
            self._current.append(f"Let t₁, t₂, ..., t_{num_free} ∈ ℝ" if num_free > 1 else "Let t₁ ∈ ℝ")
            self._current.append("")
            self._flush()

            return None  # Infinite solutions