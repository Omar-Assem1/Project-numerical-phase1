
class GaussJordan:
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
        print("        GAUSS-JORDAN ELIMINATION (RREF)")
        print("="*60)
        print("Initial Matrix:")
        for r in A:
            print("  ".join(f"{x:10.4f}" for x in r))

        for i in range(n):
            pivot_row = i
            for k in range(i, n):
                if abs(A[k][i]) > abs(A[pivot_row][i]):
                    pivot_row = k

            if abs(A[pivot_row][i]) < 1e-10:
                continue

            A[i], A[pivot_row] = A[pivot_row], A[i]

            pivot = A[i][i]
            print(f"\n→ Row {i+1} ÷ {pivot:.4f}  (make pivot = 1)")
            for j in range(n+1):
                A[i][j] /= pivot
            for r in A:
                print("  ".join(f"{x:10.4f}" for x in r))

            for k in range(n):
                if k != i and abs(A[k][i]) > 1e-10:
                    factor = A[k][i]
                    print(f"Row {k+1} -= ({factor:.4f}) × Row {i+1}")
                    for j in range(n+1):
                        A[k][j] -= factor * A[i][j]
                    for r in A:
                        print("  ".join(f"{x:10.4f}" for x in r))

        print("\n" + "="*60)
        print("            REDUCED ROW ECHELON FORM")
        print("="*60)
        for r in A:
            print("  ".join(f"{x:10.4f}" for x in r))

        rank = 0
        for row in A:
            if any(abs(x) > 1e-10 for x in row[:-1]):
                rank += 1

        for i in range(rank, n):
            if abs(A[i][n]) > 1e-10:
                print("\nNO SOLUTION → inconsistent system")
                return

        if rank < n:
            print(f"\nINFINITE SOLUTIONS (rank = {rank} < {n})")
            return

        print("\nUNIQUE SOLUTION:")
        for i in range(n):
            print(f"x{i+1} = {A[i][n]:.6f}")


n = int(input("Enter number of unknowns: "))
gj = GaussJordan(n)

print(f"Enter {n} equations (coeff + constant):")
for i in range(n):
    eq = input(f"Equation {i+1}: ")
    gj.set_equation(eq)

gj.show("Your System")
gj.solve()