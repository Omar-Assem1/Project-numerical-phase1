from decimal import Context
import sympy as sy
from sympy import Symbol


class falsePosition:

    def __init__(self, fx, xl, xu, es, imax, precision):
        self.fx = fx
        self.xl = xl
        self.xu = xu
        self.es = es
        self.imax = imax
        self.precision = precision
        self.iterations = 0
        self.step_strings = []
        self.approximateError = 0
        self.xr = None
        self.xsym = Symbol('x')

    def round_sig(self, x):
        x_str = str(x)
        try:
            ctx = Context(prec=self.precision)
            return float(ctx.create_decimal(x_str).normalize())
        except:
            return float(x)

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
            raise Exception("False Position Fails: The product of F(Xl) * F(Xu) = 0")
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
                    ea = self.round_sig(abs((xr - xr_old) / xr))
                else:
                    ea = 0

                xr_old = xr

                test_product = fxl * fxr

                if (test_product < 0):
                    xu = xr
                    fxu = fxr
                elif (test_product > 0):
                    xl = xr
                    fxl = fxr
                else:
                    ea = 0

                self.xr = xr

                self.step_strings.append(f"Step {stepCounter}, iteration {i} \n =========== \n\n "
                                         f"Xl = {xl} \n Xu = {xu} \n Root = {xr} \n Relative Error = {ea} \n fxr= {fxr}")

                stepCounter += 1
                if (ea < self.es):
                    self.step_strings.append(f"Final Result \n =========== \n\n"
                                             f" Root after iteration {i} = {xr}\n"
                                             f" \n\n Relative Error = {ea}")

                    self.approximateError = ea
                    self.iterations = i
                    return self.xr

            return self.xr

    def getSteps(self):
        return self.step_strings

    def getApproximateError(self):
        return self.approximateError

    def getIterations(self):
        return self.iterations