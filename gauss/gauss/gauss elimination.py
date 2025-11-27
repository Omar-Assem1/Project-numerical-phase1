class GaussianElimination:
    def __init__(self, n):
        self.n = n
        self.A = [[0.0 for _ in range(n + 1)] for _ in range(n)]
        self.current_row = 0

    def set_equation(self, values):
        if self.current_row >= self.n:
            print("all equations had been entered")
            return
        if isinstance(values, str):
            values = values.split()
        if len(values) != self.n + 1:
            print(f"Error: You must enter exactly {self.n + 1} numbers!")
            return
        for j in range(self.n + 1):
            self.A[self.current_row][j] = float(values[j])
        self.current_row += 1
        print(f"Equation {self.current_row} added successfully!")

    def is_complete(self):
        return self.current_row == self.n

    def show(self, title=""):
        if title:
            print(f"\n=== {title} ===")
        for r in self.A:
            print("  ".join(f"{x:10.4f}" for x in r))
        print()

    def solve(self):
        if not self.is_complete():
            print("enter all equations brother!")
            return

        A = [row[:] for row in self.A]
        n = self.n

        print("\n" + "="*60)
        print("        GAUSSIAN ELIMINATION WITH STEPS")
        print("="*60)
        print("Initial Matrix:")
        for r in A:
            print("  ".join(f"{x:10.4f}" for x in r))

        rank = 0
        for i in range(n):
            pivot_row = i
            for k in range(i, n):
                if abs(A[k][i]) > abs(A[pivot_row][i]):
                    pivot_row = k

            if abs(A[pivot_row][i]) < 1e-10:
                continue

            A[i], A[pivot_row] = A[pivot_row], A[i]
            rank += 1

            print(f"\n→ Pivot in column {i+1}, using Row {pivot_row+1} → Row {i+1}")
            for r in A:
                print("  ".join(f"{x:10.4f}" for x in r))
###############forward elimination
            for k in range(i + 1, n):
                if A[k][i] != 0:
                    factor = A[k][i] / A[i][i]
                    print(f"Row{k+1} -= ({factor:.4f}) × Row{i+1}")
                    for j in range(i, n + 1):
                        A[k][j] -= factor * A[i][j]
                    for r in A:
                        print("  ".join(f"{x:10.4f}" for x in r))

        print("\n" + "="*60)
        print("           UPPER TRIANGULAR FORM")
        print("="*60)
        for r in A:
            print("  ".join(f"{x:10.4f}" for x in r))

        for i in range(rank, n):
            if abs(A[i][n]) > 1e-10:
                print("\nNO SOLUTION (0 = non-zero) → inconsistent system")
                return

        if rank < n:
            print(f"\nINFINITE SOLUTIONS (rank = {rank} < {n} variables)")
            print("There are free variables!")
            return

        x = [0.0] * n
        for i in range(n - 1, -1, -1):
            x[i] = A[i][n]
            for j in range(i + 1, n):
                x[i] -= A[i][j] * x[j]
            x[i] /= A[i][i]

        print("\nUNIQUE SOLUTION:")
        for i in range(n):
            print(f"x{i+1} = {x[i]:.6f}")


n = int(input("Enter number of unknowns: "))
g = GaussianElimination(n)

print(f"Enter {n} equations (coeff + constant):")
for i in range(n):
    eq = input(f"Equation {i+1}: ")
    g.set_equation(eq)

g.show("Your System")
g.solve()