from decimal import Context
import sympy as sy

class bisection:

    def __init__(self, fx, xl, xu, es, imax):
        self.fx = fx
        self.xl = xl
        self.xu = xu
        self.es = es
        self.imax = imax



    def round_sig(self,x):
        ctx = Context(prec=self.precision)
        return float(ctx.create_decimal(x).normalize())

    def solve(self):
        fxl = sy.sympify(self.fx).subs("x", self.xl)
        fxu = sy.sympify(self.fx).subs("x", self.xu)
        if (fxl * fxu > 0):
            print("Bisection Fails")
            return
        else:
            xr_old = self.xl
            for i in range(1, self.imax):
                xr = (xu + xl) / 2
                fxr = sy.sympify(self.fx).subs("x", self.xr)
                ea = abs((xr - xr_old) / xr)
                xr_old = xr
                if (fxl * fxr < 0):
                    xu = xr
                elif (fxl * fxr == 0):
                    ea = 0
                else:
                    xl = xr
                if (ea < self.es):
                    return xr
