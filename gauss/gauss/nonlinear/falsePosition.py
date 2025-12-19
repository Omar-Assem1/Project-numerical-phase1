from decimal import Context
import sympy as sy
from sympy import Symbol
import math


class falsePosition:

    def __init__(self, fx, xl, xu, es=0.00001, imax=50, precision=5):
        self.fx = fx
        self.xl = xl
        self.xu = xu
        self.es = es * 100  # Convert decimal to percentage (0.00001 -> 0.001%)
        self.imax = imax
        self.precision = precision
        self.iterations = 0
        self.step_strings = []
        self.approximateError = 0
        self.xr = None
        self.xsym = Symbol('x')
        self.significant_figures = 0

    def round_sig(self, x):
        x_str = str(x)
        try:
            ctx = Context(prec=self.precision)
            return float(ctx.create_decimal(x_str).normalize())
        except:
            return float(x)

    def count_significant_figures(self, x_new, x_old):
        """Count correct significant figures."""
        if x_new == 0:
            return 0
        rel_error_decimal = abs((x_new - x_old) / x_new)
        if rel_error_decimal == 0:
            return 15
        if rel_error_decimal < 1e-10:
            return 10
        # Use percentage value in the formula: rel_error_percentage = rel_error_decimal * 100
        rel_error_percentage = rel_error_decimal * 100
        n = 2 - math.log10(2 * rel_error_percentage)
        return max(0, int(n))

    def solve(self):
        xl = self.xl
        xu = self.xu
        f_expr = sy.sympify(self.fx)

        fxl = self.round_sig(f_expr.subs({self.xsym: xl}))
        fxu = self.round_sig(f_expr.subs({self.xsym: xu}))

        stepCounter = 1

        if (fxl * fxu > 0):
            print("Regula Falsi Fails")
            self.step_strings.append(
                f"Step {stepCounter} \n =========== \n\n Regula Falsi Fails: Initial guesses do not bracket the root.")
            raise Exception("False Position Fails: The product of F(Xl)*F(Xu) > 0")
        elif (fxl * fxu == 0):
            # One of the bounds is exactly the root
            if fxl == 0:
                self.xr = xl
                self.approximateError = 0
                self.iterations = 0
                self.significant_figures = self.precision
                self.step_strings.append(f"Step {stepCounter} \n =========== \n\n "
                                         f"Exact root found at lower bound: xl = {xl} \n f(xl) = {fxl}")
                return xl
            else:  # fxu == 0
                self.xr = xu
                self.approximateError = 0
                self.iterations = 0
                self.significant_figures = self.precision
                self.step_strings.append(f"Step {stepCounter} \n =========== \n\n "
                                         f"Exact root found at upper bound: xu = {xu} \n f(xu) = {fxu}")
                return xu
        else:
            xr_old = xl

            for i in range(1, self.imax + 1):
                numerator = (xl * fxu) - (xu * fxl)
                denominator = fxu - fxl

                if denominator == 0:
                    raise Exception("Division by zero: f(xu) equals f(xl)")

                xr = self.round_sig(numerator / denominator)
                fxr = self.round_sig(f_expr.subs({self.xsym: xr}))

                if xr != 0:
                    ea = self.round_sig(abs((xr - xr_old) / xr) * 100)  # Convert to percentage
                else:
                    ea = 0

                test_product = fxl * fxr

                if (test_product < 0):
                    xu = xr
                    fxu = fxr
                elif (test_product > 0):
                    xl = xr
                    fxl = fxr
                else:
                    ea = 0

                xr_old = xr
                self.xr = xr

                self.step_strings.append(f"Step {stepCounter}, iteration {i} \n =========== \n\n "
                                         f"Xl = {xl} \n Xu = {xu} \n Root = {xr} \n Relative Error = {ea}% \n fxr= {fxr}")

                stepCounter += 1
                
                # Simple convergence check - stop when error <= epsilon OR max iterations
                if (ea <= self.es):
                    self.step_strings.append(f"Final Result \n =========== \n\n"
                                             f" Root after iteration {i} = {xr}\n"
                                             f" \n\n Relative Error = {ea}%")

                    self.approximateError = ea
                    self.iterations = i
                    
                    # Calculate significant figures at the END
                    if ea == 0:
                        self.significant_figures = self.precision  # Use precision when exact solution found
                    else:
                        # Use the actual relative error percentage that was calculated
                        if ea < 1e-10:
                            self.significant_figures = 10
                        else:
                            n = 2 - math.log10(2 * ea)  # ea is already in percentage
                            self.significant_figures = max(0, int(n))
                    
                    return self.xr

            # Max iterations reached without convergence
            self.approximateError = ea
            self.iterations = self.imax
            
            # Calculate significant figures at the END
            if ea == 0:
                self.significant_figures = self.precision  # Use precision when exact solution found
            else:
                # Use the actual relative error percentage that was calculated
                if ea < 1e-10:
                    self.significant_figures = 10
                else:
                    n = 2 - math.log10(2 * ea)  # ea is already in percentage
                    self.significant_figures = max(0, int(n))
            
            return self.xr

    def getSteps(self):
        return self.step_strings

    def getApproximateError(self):
        return self.approximateError

    def getIterations(self):
        return self.iterations

    def getSignificantFigures(self):
        return self.significant_figures