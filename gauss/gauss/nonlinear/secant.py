import numpy as np
import sympy as sy
from sympy import Symbol
from decimal import Context, Decimal


class Secant:
    def __init__(self, f, x0, x1, tol, maxiter, precision):
        self.f_expr = sy.sympify(f)
        self.x0 = x0
        self.x1 = x1
        self.tol = tol
        self.maxiter = maxiter
        self.precision = precision
        self.xsym = Symbol('x')
        self.step_strings = []
        self.iterations = 0
        self.approximateError = None
        self.converged = False
        self.root = None

    def round_sig(self, x):
        """Round number to specified precision using significant figures."""
        if x == 0:
            return 0.0
        # Check for special values
        if np.isnan(x) or np.isinf(x):
            return x
        try:
            x_str = str(x)
            ctx = Context(prec=self.precision)
            return float(ctx.create_decimal(x_str).normalize())
        except:
            return x

    def relative_error(self, x_new, x_old):
        """Calculate relative error between successive iterations."""
        if x_new == 0:
            return float('inf')
        return self.round_sig(abs(x_new - x_old) / abs(x_new))

    def solve(self):
        """Solve using Secant method and return the root."""
        x0 = self.round_sig(self.x0)
        x1 = self.round_sig(self.x1)

        self.step_strings.append(
            "Secant Method Formula: x_{i+1} = x_i - f(x_i) * (x_i - x_{i-1}) / (f(x_i) - f(x_{i-1}))\n"
        )

        for i in range(self.maxiter):
            # Evaluate function at current points
            f0 = self.round_sig(float(self.f_expr.subs(self.xsym, x0)))
            f1 = self.round_sig(float(self.f_expr.subs(self.xsym, x1)))

            # Check for division by zero
            if f1 == f0:
                self.step_strings.append(
                    f"\nIteration {i}: Method failed - f(x_i) = f(x_{{i-1}}) = {f1}"
                )
                self.step_strings.append("Cannot continue: denominator is zero")
                return None

            # Calculate next approximation
            x_new = x1 - f1 * (x1 - x0) / (f1 - f0)
            x_new = self.round_sig(x_new)

            # Calculate relative error
            re = self.relative_error(x_new, x1)

            # Store iteration details
            step = (
                f"\nIteration {i}:\n"
                f"  x_{{i-1}} = {x0}\n"
                f"  x_i = {x1}\n"
                f"  f(x_{{i-1}}) = {f0}\n"
                f"  f(x_i) = {f1}\n"
                f"  x_{{i+1}} = {x1} - {f1} * ({x1} - {x0}) / ({f1} - {f0})\n"
                f"  x_{{i+1}} = {x_new}\n"
                f"  Relative Error = |{x_new} - {x1}| / |{x_new}| = {re}"
            )
            self.step_strings.append(step)

            self.iterations = i + 1
            self.approximateError = re

            # Check convergence
            if re < self.tol:
                self.step_strings.append(
                    f"\n✓ Convergence achieved!\n"
                    f"Root found: x = {x_new}\n"
                    f"Iterations: {self.iterations}\n"
                    f"Final relative error: {re}"
                )
                self.root = x_new
                self.converged = True
                return x_new

            # Update for next iteration
            x0 = x1
            x1 = x_new

        # Max iterations reached
        self.step_strings.append(
            f"\n✗ Maximum iterations ({self.maxiter}) reached without convergence.\n"
            f"Last approximation: x = {x1}\n"
            f"Final relative error: {re}"
        )
        self.root = x1
        return x1

    def get_answer(self):
        """Return step strings for backward compatibility."""
        return self.step_strings

    def printSteps(self):
        """Print all solution steps."""
        for line in self.step_strings:
            print(line)
