from io import BytesIO
import base64
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import math


def get_plot_base64(function: str, include_yx_plot: bool = False):

    safe_env = {
        "sin": math.sin, "cos": math.cos, "tan": math.tan,
        "sqrt": math.sqrt, "log": math.log, "exp": math.exp,
        "pi": math.pi, "e": math.e,
        "abs": abs, "pow": pow
    }

    def user_func(x_val):
        try:
            return eval(
                function,
                {"__builtins__": None},
                {**safe_env, "x": x_val},
            )
        except:
            return float("nan")

    x_range = np.linspace(-10, 10, 400)
    y_range = [user_func(x) for x in x_range]

    fig = Figure(figsize=(10, 6), dpi=120)
    FigureCanvasAgg(fig)
    ax = fig.subplots()

    ax.plot(x_range, y_range, label=f"y = {function}", color="blue")

    if include_yx_plot:
        ax.plot(x_range, x_range, label="y = x", color="red", linestyle="--")

    ax.spines["left"].set_position("center")
    ax.spines["bottom"].set_position("center")
    ax.spines["right"].set_color("none")
    ax.spines["top"].set_color("none")
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")

    ax.set_xticks(np.arange(-10, 11, 1))
    ax.set_yticks(np.arange(-30, 31, 1))

    ax.xaxis.set_ticklabels([t if t % 2 == 0 else "" for t in range(-10, 11)])
    ax.yaxis.set_ticklabels([t if t % 5 == 0 else "" for t in range(-30, 31)])

    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="upper left")

    ax.set_xlim(-10, 10)
    ax.set_ylim(-30, 30)

    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    data = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{data}"