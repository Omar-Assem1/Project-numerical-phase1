import math
import sympy as sy
from sympy import Symbol
from decimal import Context, Decimal  # FIXED

class Secant():
  def __init__(self,f,x0,x1,tol,maxiter,precision):
    self.f_expr = sy.sympify(f)
    self.x0=x0
    self.x1=x1
    self.tol=tol
    self.maxiter=maxiter
    self.precision=precision
    self.xsym = Symbol('x')
    self.answer = []

  def get_answer(self):
    return self.answer

  def relative_error(self,x_new,x_old):
    return self.round_sig(abs(x_new-x_old)/abs(x_new))

  def round_sig(self,x):
    x_str = str(x)
    ctx = Context(prec=self.precision)
    return float(ctx.create_decimal(x_str).normalize())

  def solve(self):
    x0 = self.round_sig(self.x0)
    x1 = self.round_sig(self.x1)
    self.answer.append("Secant Method: Xi+1 = xi - f(Xi) * ( Xi - Xi-1 ) / ( f(Xi) - f(Xi-1) )")
    for i in range(self.maxiter):
      # Use dictionary syntax for substitution: {Symbol: value}
      f0 = self.round_sig(float(self.f_expr.subs(self.xsym, x0)))
      f1 = self.round_sig(float(self.f_expr.subs(self.xsym, x1)))
      # avoid divide-by-zero
      if f1 == f0:
        self.answer.append("Method can't be applied (f(xi) == f(xi-1))")
        return None

      x_new = x1 - f1 * (x1 - x0) / (f1 - f0)
      x_new = self.round_sig(x_new)
      re = self.relative_error(x_new, x1)
      ans = (
                f"Iteration: {i}  "
                f"Xi-1: {x0}  "
                f"Xi: {x1}  "
                f"Xi+1: {x_new}  "
                f"f(Xi-1): {f0}  "
                f"f(Xi): {f1}  "
                f"Relative error: {re}"
            )
      self.answer.append(ans)
      if re < self.tol:
        self.answer.append(f"Root: {x_new} found after {i} iterations")
        return x_new

      x0 = x1
      x1 = x_new
    self.answer.append(f"Root not found after {self.maxiter} iterations")
    return None

  def printSteps(self):
    for line in self.answer:
      print(line)
