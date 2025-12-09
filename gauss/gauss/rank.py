import copy


class SolutionType:
    INCONSISTENT = 1
    INFINITE = 2
    UNIQUE = 3

    def __init__(self, augmented_matrix, tol=1e-10):
        self.A = [row[:-1] for row in augmented_matrix]  # coefficient part
        self.b = [row[-1] for row in augmented_matrix]  # right-hand side
        self.n = len(augmented_matrix)
        self.M = [row[:] for row in augmented_matrix]  # working copy
        self.tol = tol

    def _swap_rows(self, i, j):
        if i != j:
            self.M[i], self.M[j] = self.M[j], self.M[i]

    def _find_pivot(self, col, start_row):
        best_idx = None
        best_val = 0
        for i in range(start_row, self.n):
            val = abs(self.M[i][col])
            if val > self.tol and (best_idx is None or val > best_val):
                best_val = val
                best_idx = i
        return best_idx

    def gaussian_elimination(self):
        row = 0  # current row
        pivot_cols = set()

        for col in range(self.n):
            # Step 1: Find pivot
            pivot_row = self._find_pivot(col, row)
            if pivot_row is None:
                continue

            # Step 2: Swap to bring pivot to position
            self._swap_rows(row, pivot_row)

            # Step 3: Eliminate below
            for i in range(row + 1, self.n):
                if abs(self.M[i][col]) < self.tol:
                    continue
                factor = self.M[i][col] / self.M[row][col]
                for j in range(col, self.n + 1):
                    self.M[i][j] -= factor * self.M[row][j]

            pivot_cols.add(col)
            row += 1
            if row >= self.n:
                break

        rank = row

        # Now analyze the reduced form
        # Check rows from rank onward (should be all zeros in coefficients)
        for i in range(rank, self.n):
            # If coefficient row is zero but RHS is not → inconsistent
            if abs(self.M[i][self.n]) > self.tol:
                return self.INCONSISTENT

        # If rank < n → free variables → infinite solutions
        if rank < self.n:
            return self.INFINITE

        # Otherwise: full rank square system and consistent → unique
        return self.UNIQUE