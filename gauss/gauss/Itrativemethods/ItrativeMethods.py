import math
from decimal import Context

class ItrativeMethods:

    def __init__(self, n, mat, ansV, X, it, tol, precision):
        self.__n = n
        self.__mat = mat
        self.__ansV = ansV
        self.__X = X
        self.__it = it
        self.__tol = tol
        self.__precision = precision
        self.__numberOfIterations = 0
        self.__converge = False
        self.__answer = []

    def setMatrix(self, mat, ansV):
        self.__mat = mat
        self_ansV = ansV

    def setIt(self, it):
        self.__it = it

    def setTol(self, tol):
        self.__tol = tol

    def __setConvergence(self, converge):
        self.__converge = converge

    def __setNumberOfIterations(self, numberOfIterations):
        self.__numberOfIterations = numberOfIterations

    def getMatrix(self):
        return self.__mat

    def getIterations(self):
        return self.__it

    def getTolerance(self):
        return self.__tol

    def getConvergence(self):
        return self.__converge

    def getNumberOfIterations(self):
        return self.__numberOfIterations

    def getAnswer(self):
        return self.__answer

    def reset(self):
        self.__numberOfIterations = 0
        self.__converge = False
        self.__answer = []

    def error(self, X, X_new):
        
        errors = [abs((X_new[i] - X[i]) / max(X_new[i],1e-10)) for i in range(len(X))]
        return errors

    def round_sig(self, x):
        # Create a context with the desired precision
        ctx = Context(prec=self.__precision)
        # Normalize applies the precision to the number
        return float(ctx.create_decimal(x).normalize())

    def _check_diagonal_dominance(self):
        ans = "Diagonal dominance check:\n"
        dominant = True
        for i in range(self.__n):
            diag = abs(self.__mat[i][i])
            off = sum(abs(self.__mat[i][j]) for j in range(self.__n) if j != i)
            status = "STRICT" if diag > off else "WEAK" if diag >= off else "NO"
            if diag <= off:
                dominant = False
            ans += (f"  Row {i + 1}: |a{i + 1}{i + 1}| = {diag:.6g} vs Σ|others| = {off:.6g} → {status}\n")
        ans += ("→ Strictly diagonally dominant → convergence guaranteed!\n" if dominant else
                "→ Not strictly dominant → convergence not guaranteed\n")
        self.__answer.append(ans)

    def print_iteration_formulas(self, method):
        n = self.__n
        ans = (f"Formulas for {method} method:\n")
        for i in range(n):
            formula = f"x{i + 1}(new) = "
            s = self.__ansV[i]
            terms = []
            for j in range(n):
                if i == j:
                    continue
                coef = self.__mat[i][j]
                var_type = "old" if method == "jacobi" else ("new" if j < i else "old")
                terms.append(f"{'-' if coef < 0 else '+'} {abs(coef)}x{j + 1}({var_type})")
            formula += f"({s} {' '.join(terms)}) / {self.__mat[i][i]}\n"
            ans += formula
        self.__answer.append(ans)

    def jacobi(self):
        n = self.__n
        X = self.__X.copy()
        X_new = [0.0] * n  # jacobi

        for i in range(n):
            if self.__mat[i][i] == 0:
                self.__answer.append(f"this method can't be used A[{i}][{i}] == zero")
                return

        self._check_diagonal_dominance()

        for iterations in range(self.__it):
            self.__numberOfIterations = iterations + 1
            X_old = X.copy()
            for i in range(n):
                calculations = self.round_sig(self.__ansV[i])
                for j in range(n):
                    if i == j: continue
                    sub_value = self.round_sig(self.__mat[i][j] * X[j])
                    calculations -= sub_value
                    calculations = self.round_sig(calculations)
                X_new[i] = self.round_sig(calculations / self.__mat[i][i])  # jacobi
            X = X_new.copy()  # jacobi

            ans = f"iteration {self.__numberOfIterations}: {X}\n"
            e = self.error(X_old, X)
            ans += f"relative error: {e}\n"
            self.__answer.append(ans)
            if max(e) < self.__tol:
                self.__converge = True
                ans = f"Jacobi Solution: {X}\n"
                ans += f"Converged: {self.getConvergence()}, Iterations: {self.getNumberOfIterations()}\n"
                self.__answer.append(ans)
                break
        ans = f"Jacobi Solution: {X}\n"
        ans += f"Converged: {self.getConvergence()}, Iterations: {self.getNumberOfIterations()}\n"
        self.__answer.append(ans)
        return X

    def seidel(self):

        n = self.__n
        X = self.__X.copy()

        for i in range(n):
            if self.__mat[i][i] == 0:
                self.__answer.append(f"this method can't be used A[{i}][{i}] == zero\n")
                return

        self._check_diagonal_dominance()

        for iterations in range(self.__it):
            self.__numberOfIterations = iterations + 1
            X_old = X.copy()
            for i in range(n):
                calculations = self.round_sig(self.__ansV[i])
                for j in range(n):
                    if i == j: continue
                    sub_value = self.round_sig(self.__mat[i][j] * X[j])
                    calculations -= sub_value
                    calculations = self.round_sig(calculations)
                X[i] = self.round_sig(calculations / self.__mat[i][i])  # seidel

            ans = f"iteration {self.__numberOfIterations}: {X}\n"
            e = self.error(X_old, X)
            ans += f"relative error: {e}\n"
            self.__answer.append(ans)
            if max(e) <self.__tol:
                self.__converge = True
                ans = f"Gauss-Seidel Solution: {X}\n"
                ans += f"Converged: {self.getConvergence()}, Iterations: {self.getNumberOfIterations()}\n"
                self.__answer.append(ans)
                break
        ans = f"Gauss-Seidel Solution: {X}\n"
        ans += f"Converged: {self.getConvergence()}, Iterations: {self.getNumberOfIterations()}\n"
        self.__answer.append(ans)
        return X

    def symbolic_iterations(self, iterations, method="jacobi"):
        import sympy as sp

        n = self.__n

        X = sp.symbols("x1:%d" % (n + 1))

        A = [[sp.sympify(self.__mat[i][j]) for j in range(n)] for i in range(n)]
        b = [sp.sympify(self.__ansV[i]) for i in range(n)]

        for i in range(n):
            if A[i][i] == 0:
                print(f"this method can't be used A[{i}][{i}] == zero\n")
                return
        X_current = list(X)

        for _ in range(iterations):
            X_new = [None] * n
            for i in range(n):
                expr = b[i]
                for j in range(n):
                    if i == j:
                        continue
                    if method == "jacobi":
                        expr -= A[i][j] * X_current[j]
                    else:
                        expr -= A[i][j] * (X_new[j] if j < i else X_current[j])

                X_new[i] = sp.simplify(expr / A[i][i])

            X_current = X_new.copy()

        return X_current

#
# A = [
#     [4, -1, 0],
#     [-1, 0, -1],
#     [0, -1, 3]
# ]
# b = [15, 10, 10]
# X = [1, 1, 1]
#
# solver = ItrativeMethods(3, A, b, X, it=25, tol=1e-4, precision=4)
#
# print("===== Jacobi Method =====")
# solver.print_iteration_formulas("jacobi")
# X_jacobi = solver.jacobi()
# for line in solver.getAnswer():
#     print(line, end='')
#
# solver.reset()
#
# print("\n===== Gauss-Seidel Method =====")
# solver.print_iteration_formulas("seidel")
# X_seidel = solver.seidel()
# for line in solver.getAnswer():
#     print(line, end='')
#
# # A = [["a","b","c"],
# #      ["d","e","f"],
# #      ["g","h","i"]]
#
# # b = ["j","k","l"]
# # X = ["x1","x2","x3"]
#
# # solver = ItrativeMethods(3, A, b, X, it=5, tol=1e-4, precision=4)
#
# # print("Jacobi:", solver.symbolic_iterations(2, method="jacobi"))
# # print("Gauss-Seidel:", solver.symbolic_iterations(2, method="seidel"))

