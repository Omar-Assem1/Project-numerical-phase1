from io import BytesIO
import base64
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import math


def get_plot_base64(function: str, include_yx_plot: bool = False):
    """
    Generates a matplotlib plot of a user-defined string function
    and optionally plots the y=x line, returning a base64 URI.
    """
    print(function)
    # Safe math environment
    safe_env = {
        "sin": math.sin, "cos": math.cos, "tan": math.tan,
        "sqrt": math.sqrt, "log": math.log, "exp": math.exp,
        "pi": math.pi, "e": math.e,
        "abs": abs, "pow": pow,
    }

    def user_func(x_val):
        local_env = {"x": x_val}
        try:
            return eval(function, {"__builtins__": None}, {**safe_env, **local_env})
        except Exception:
            return float('nan')

    # Data
    x_range = np.linspace(-10, 10, 400)
    y_range = [user_func(x) for x in x_range]

    # Create figure + attach renderer backend
    fig = Figure(figsize=(6, 4))
    FigureCanvasAgg(fig)     # <-- REQUIRED
    ax = fig.subplots()

    # Plot function
    ax.plot(x_range, y_range, label=f"y = {function}", color='blue')

    if include_yx_plot:
        ax.plot(x_range, x_range, label="y = x", color='red', linestyle='--')

    ax.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax.axvline(0, color='gray', linestyle='--', linewidth=1)
    ax.legend()
    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    ax.set_title("Function Plot")
    ax.grid(True)
    ax.set_ylim(-3, 3)
    ax.set_xlim(-2, 2)

    # Convert to base64
    buf = BytesIO()
    fig.savefig(buf, format="png")
    data = base64.b64encode(buf.getvalue()).decode("ascii")

    return f"data:image/png;base64,{data}"