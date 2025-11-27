# classes/linear_system.py
class LinearSystem:
    def __init__(self, n: int , precision = None , tol = None):
        self.n = n
        self.A = [[0.0] * (n + 1) for _ in range(n)]
        self.precision = precision if precision is not None else 4
        self.tolerence = tol if tol is not None else 1e-20
        self.current_row = 0

    def add_equation(self, values):
        if self.current_row >= self.n:
            print("All equations already entered!")
            return False

        if isinstance(values, str):
            values = values.replace(',', ' ').split()

        if len(values) != self.n + 1:
            print(f"Error: Need {self.n + 1} numbers (coefficients + constant)")
            return False

        try:
            self.A[self.current_row] = [float(x) for x in values]
            self.current_row += 1
            print(f"Equation {self.current_row} added successfully!")
            return True
        except ValueError:
            print("Error: Invalid number format!")
            return False

    def is_complete(self) -> bool:
        return self.current_row == self.n

    def copy_matrix(self):
        """Return a deep copy of the augmented matrix"""
        return [row[:] for row in self.A]

    def show(self, title: str = "Augmented Matrix"):
        print(f"\n=== {title} ===")
        for i, row in enumerate(self.A):
            print(f"R{i+1}:", "  ".join(f"{x:12.4f}" for x in row))
        print()