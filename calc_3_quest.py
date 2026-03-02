import streamlit as st
import numpy as np
import sympy as sp
import plotly.graph_objects as go
from dataclasses import dataclass
from typing import Callable, Dict, Any, Tuple, Optional

# -----------------------------
# Streamlit config
# -----------------------------
st.set_page_config(page_title="Calc 3 Quest", page_icon="🧭", layout="centered")

# -----------------------------
# Sympy setup
# -----------------------------
x, y, z, t = sp.symbols("x y z t", real=True)

# -----------------------------
# Rendering helpers (FIXES ALL FORMATTING)
# -----------------------------
def md(s: str):
    st.markdown(s)

def tex_inline(s: str) -> str:
    return f"${s}$"

def tex_block(s: str):
    st.latex(s)

def hr():
    st.divider()

def badge(msg: str):
    st.success(msg)

def fail(msg: str):
    st.error(msg)

def hint(msg: str):
    st.info(msg)

def note(msg: str):
    st.warning(msg)

# -----------------------------
# Math parsing helpers
# -----------------------------
def safe_sympify(expr: str):
    locals_dict = {
        "pi": sp.pi,
        "sqrt": sp.sqrt,
        "sin": sp.sin,
        "cos": sp.cos,
        "tan": sp.tan,
        "exp": sp.exp,
        "ln": sp.log,
        "log": sp.log,
        "abs": sp.Abs,
    }
    return sp.sympify(expr, locals=locals_dict)

def parse_vec(expr: str) -> sp.Matrix:
    """
    Accepts: [1,2,3]  or <1,2,3> or (1,2,3)
    """
    s = expr.strip().replace("<", "[").replace(">", "]").replace("(", "[").replace(")", "]")
    v = safe_sympify(s)
    if isinstance(v, (list, tuple)):
        v = sp.Matrix(v)
    if isinstance(v, (sp.Matrix, sp.ImmutableMatrix)) and v.shape == (3, 1):
        return v
    if isinstance(v, (sp.Matrix, sp.ImmutableMatrix)) and v.shape == (1, 3):
        return v.T
    raise ValueError("Could not parse a 3D vector. Try format like [1,2,3].")

def format_vec(v: sp.Matrix) -> str:
    return f"\\langle {sp.simplify(v[0])}, {sp.simplify(v[1])}, {sp.simplify(v[2])} \\rangle"

def almost_equal(a, b, tol=1e-6):
    try:
        return abs(float(a) - float(b)) <= tol
    except Exception:
        return False

# -----------------------------
# Plotly visual helpers
# -----------------------------
def plot_vectors_3d(v: np.ndarray, w: np.ndarray, title="3D Vectors"):
    # Draw vectors from origin + a faint grid cube.
    fig = go.Figure()

    def add_vec(vec, name):
        fig.add_trace(
            go.Scatter3d(
                x=[0, vec[0]],
                y=[0, vec[1]],
                z=[0, vec[2]],
                mode="lines+markers",
                name=name,
                marker=dict(size=4),
            )
        )

    add_vec(v, "v")
    add_vec(w, "w")

    # show axes lines a bit
    axis_len = max(1, int(max(np.linalg.norm(v), np.linalg.norm(w))) + 1)
    fig.add_trace(go.Scatter3d(x=[-axis_len, axis_len], y=[0, 0], z=[0, 0], mode="lines", name="x-axis", opacity=0.25))
    fig.add_trace(go.Scatter3d(x=[0, 0], y=[-axis_len, axis_len], z=[0, 0], mode="lines", name="y-axis", opacity=0.25))
    fig.add_trace(go.Scatter3d(x=[0, 0], y=[0, 0], z=[-axis_len, axis_len], mode="lines", name="z-axis", opacity=0.25))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="x",
            yaxis_title="y",
            zaxis_title="z",
            aspectmode="cube",
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        showlegend=True,
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_plane_and_point(n: np.ndarray, d: float, P: np.ndarray, title="Plane + Point"):
    # Plane: n·[x,y,z] = d. We'll solve for z if possible, otherwise for y, otherwise x.
    # Create a mesh grid and compute surface.
    a, b, c = n.tolist()
    fig = go.Figure()

    rng = 4
    grid = np.linspace(-rng, rng, 25)
    X, Y = np.meshgrid(grid, grid)

    surface_added = False
    if abs(c) > 1e-8:
        Z = (d - a * X - b * Y) / c
        fig.add_trace(go.Surface(x=X, y=Y, z=Z, opacity=0.5, name="plane", showscale=False))
        surface_added = True
    elif abs(b) > 1e-8:
        # solve for y = (d - a x - c z)/b with x,z grid
        X2, Z2 = np.meshgrid(grid, grid)
        Y2 = (d - a * X2 - c * Z2) / b
        fig.add_trace(go.Surface(x=X2, y=Y2, z=Z2, opacity=0.5, name="plane", showscale=False))
        surface_added = True
    elif abs(a) > 1e-8:
        # solve for x = (d - b y - c z)/a with y,z grid
        Y2, Z2 = np.meshgrid(grid, grid)
        X2 = (d - b * Y2 - c * Z2) / a
        fig.add_trace(go.Surface(x=X2, y=Y2, z=Z2, opacity=0.5, name="plane", showscale=False))
        surface_added = True

    # point
    fig.add_trace(go.Scatter3d(x=[P[0]], y=[P[1]], z=[P[2]], mode="markers+text", text=["P"], textposition="top center", name="point"))

    # normal vector from plane point (closest-ish point just use origin projection-ish for display)
    base = np.array([0.0, 0.0, 0.0])
    n_unit = n / (np.linalg.norm(n) + 1e-12)
    fig.add_trace(go.Scatter3d(
        x=[base[0], base[0] + n_unit[0]*2],
        y=[base[1], base[1] + n_unit[1]*2],
        z=[base[2], base[2] + n_unit[2]*2],
        mode="lines+markers",
        name="normal (direction)",
        opacity=0.8
    ))

    if not surface_added:
        # Shouldn't happen unless n ~ 0
        pass

    fig.update_layout(
        title=title,
        scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z", aspectmode="cube"),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_curve_r_t():
    # r(t) = <t, 3cos t, 3 sin t>
    ts = np.linspace(-5, 5, 400)
    X = ts
    Y = 3*np.cos(ts)
    Z = 3*np.sin(ts)
    fig = go.Figure(go.Scatter3d(x=X, y=Y, z=Z, mode="lines", name="r(t)"))
    fig.update_layout(
        title="Curve: r(t) = <t, 3cos t, 3sin t>",
        scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z", aspectmode="cube"),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_trace_circle_z_equals_1():
    # z = x^2 + y^2, trace z=1 => circle x^2 + y^2 = 1 in plane z=1
    theta = np.linspace(0, 2*np.pi, 300)
    X = np.cos(theta)
    Y = np.sin(theta)
    Z = np.ones_like(theta) * 1.0
    fig = go.Figure(go.Scatter3d(x=X, y=Y, z=Z, mode="lines", name="trace z=1"))
    fig.update_layout(
        title="Trace of z = x^2 + y^2 at z=1  →  x^2 + y^2 = 1",
        scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z", aspectmode="cube"),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Game structure
# -----------------------------
@dataclass
class Level:
    id: str
    title: str
    teach_render: Callable[[], None]
    mission_render: Callable[[], None]
    ui: Callable[[], Dict[str, Any]]
    check: Callable[[Dict[str, Any]], Tuple[bool, str]]
    viz: Optional[Callable[[Dict[str, Any]], None]] = None

def add_xp(base: int):
    st.session_state.xp += base

def reset_attempts():
    st.session_state.attempts = 0

# -----------------------------
# Session state
# -----------------------------
if "level_idx" not in st.session_state:
    st.session_state.level_idx = 0
if "xp" not in st.session_state:
    st.session_state.xp = 0
if "lives" not in st.session_state:
    st.session_state.lives = 3
if "attempts" not in st.session_state:
    st.session_state.attempts = 0
if "completed" not in st.session_state:
    st.session_state.completed = set()
if "practice_mode" not in st.session_state:
    st.session_state.practice_mode = False

# -----------------------------
# Level 1: Dot Product
# -----------------------------
def L1_teach():
    md("You’ll use vectors constantly in Calc 3.")
    md("### Dot product")
    md("Dot product measures how aligned two vectors are.")
    md("If")
    tex_block(r"\vec v=\langle v_1,v_2,v_3\rangle \quad \text{and} \quad \vec w=\langle w_1,w_2,w_3\rangle")
    md("then")
    tex_block(r"\vec v\cdot \vec w = v_1w_1+v_2w_2+v_3w_3")
    md("If " + tex_inline(r"\vec v\cdot \vec w=0") + ", they’re perpendicular.")
    md("**Goal:** compute a dot product accurately.")

def L1_mission():
    md("### Mission")
    md("Compute " + tex_inline(r"\vec v\cdot \vec w") + " for "
       + tex_inline(r"\vec v=\langle 1,2,3\rangle")
       + " and "
       + tex_inline(r"\vec w=\langle 4,0,-1\rangle") + ".")

def L1_ui():
    md("Enter vectors as `[a,b,c]` (or `<a,b,c>`).")
    v_in = st.text_input("v =", value="[1,2,3]", key="L1_v")
    w_in = st.text_input("w =", value="[4,0,-1]", key="L1_w")
    ans = st.text_input("Your dot product value =", value="", key="L1_ans")
    return {"v_in": v_in, "w_in": w_in, "ans": ans}

def L1_check(data):
    try:
        v = parse_vec(data["v_in"])
        w = parse_vec(data["w_in"])
        target = sp.simplify(v.dot(w))
        user = safe_sympify(data["ans"])
        if sp.simplify(user - target) == 0:
            return True, f"Correct: v·w = {target}."
        return False, "Not quite. Multiply matching components and add: v1w1 + v2w2 + v3w3."
    except Exception as e:
        return False, f"Parse error: {e}"

def L1_viz(data):
    try:
        v = np.array([float(sp.N(val)) for val in parse_vec(data["v_in"])], dtype=float)
        w = np.array([float(sp.N(val)) for val in parse_vec(data["w_in"])], dtype=float)
        plot_vectors_3d(v, w, title="Vectors v and w (from origin)")
        dp = float(np.dot(v, w))
        md(f"Dot product value (computed): **{dp:.4g}**")
    except Exception:
        pass

# -----------------------------
# Level 2: Cross Product
# -----------------------------
def L2_teach():
    md("### Cross product")
    md("The cross product produces a vector perpendicular to both inputs.")
    tex_block(r"\vec v\times \vec w=\begin{vmatrix}\hat i & \hat j & \hat k\\ v_1&v_2&v_3\\ w_1&w_2&w_3\end{vmatrix}")
    md("Key facts:")
    md("- Direction: perpendicular to both (right-hand rule).")
    md("- Magnitude: " + tex_inline(r"|\vec v\times \vec w|") + " is the parallelogram area spanned by v and w.")
    md("**Goal:** compute a cross product.")

def L2_mission():
    md("### Mission")
    md("Compute " + tex_inline(r"\vec v\times \vec w")
       + " for " + tex_inline(r"\vec v=\langle 1,2,0\rangle")
       + " and " + tex_inline(r"\vec w=\langle 0,1,3\rangle") + ".")

def L2_ui():
    v_in = st.text_input("v =", value="[1,2,0]", key="L2_v")
    w_in = st.text_input("w =", value="[0,1,3]", key="L2_w")
    ans = st.text_input("Your v×w =", value="[0,0,0]", key="L2_ans")
    return {"v_in": v_in, "w_in": w_in, "ans": ans}

def L2_check(data):
    try:
        v = parse_vec(data["v_in"])
        w = parse_vec(data["w_in"])
        target = sp.simplify(v.cross(w))
        user = parse_vec(data["ans"])
        if all(sp.simplify(user[i] - target[i]) == 0 for i in range(3)):
            area = sp.simplify(sp.sqrt(target.dot(target)))
            return True, f"Correct: v×w = {sp.simplify(target.T)} and |v×w| = {area} (area)."
        return False, "Not quite. Try the determinant method and watch the minus sign in the j-component."
    except Exception as e:
        return False, f"Error: {e}"

def L2_viz(data):
    try:
        v = np.array([float(sp.N(val)) for val in parse_vec(data["v_in"])], dtype=float)
        w = np.array([float(sp.N(val)) for val in parse_vec(data["w_in"])], dtype=float)
        c = np.cross(v, w)
        fig = go.Figure()
        for vec, name in [(v, "v"), (w, "w"), (c, "v×w")]:
            fig.add_trace(go.Scatter3d(
                x=[0, vec[0]], y=[0, vec[1]], z=[0, vec[2]],
                mode="lines+markers", name=name, marker=dict(size=4)
            ))
        fig.update_layout(
            title="v, w, and v×w (perpendicular)",
            scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z", aspectmode="cube"),
            margin=dict(l=0, r=0, t=40, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)
        md(f"Area magnitude |v×w| = **{np.linalg.norm(c):.4g}**")
    except Exception:
        pass

# -----------------------------
# Level 3: Plane from normal + point
# -----------------------------
def L3_teach():
    md("### Planes from a normal vector")
    md("If a plane has a normal vector " + tex_inline(r"\vec n=\langle a,b,c\rangle")
       + " and passes through " + tex_inline(r"P_0=(x_0,y_0,z_0)") + ", then:")
    tex_block(r"\vec n\cdot\langle x-x_0, y-y_0, z-z_0\rangle = 0")
    md("This expands to the standard form:")
    tex_block(r"ax+by+cz = d \quad \text{where } d=ax_0+by_0+cz_0")
    md("**Goal:** build the plane equation quickly.")

def L3_mission():
    md("### Mission")
    md("For " + tex_inline(r"\vec n=\langle 2,-1,3\rangle")
       + " and " + tex_inline(r"P_0=(1,0,2)") + ", find d in:")
    tex_block(r"2x - y + 3z = d")

def L3_ui():
    n_in = st.text_input("Normal n =", value="[2,-1,3]", key="L3_n")
    P0_in = st.text_input("Point P0 =", value="[1,0,2]", key="L3_P0")
    d_in = st.text_input("Enter d =", value="", key="L3_d")
    return {"n_in": n_in, "P0_in": P0_in, "d_in": d_in}

def L3_check(data):
    try:
        n = parse_vec(data["n_in"])
        P0 = parse_vec(data["P0_in"])
        target = sp.simplify(n.dot(P0))
        user = safe_sympify(data["d_in"])
        if sp.simplify(user - target) == 0:
            return True, f"Correct: d = {target}."
        return False, "Not quite. Compute d by plugging the point into ax+by+cz."
    except Exception as e:
        return False, f"Error: {e}"

def L3_viz(data):
    try:
        n = parse_vec(data["n_in"])
        P0 = parse_vec(data["P0_in"])
        d_val = float(sp.N(n.dot(P0)))
        n_np = np.array([float(sp.N(n[i])) for i in range(3)], dtype=float)
        P_np = np.array([float(sp.N(P0[i])) for i in range(3)], dtype=float)
        plot_plane_and_point(n_np, d_val, P_np, title="Plane (from n·x=d) and the point P0")
    except Exception:
        pass

# -----------------------------
# Level 4: Distances in 3D
# -----------------------------
def L4_teach():
    md("### Distances in 3D")
    md("Two super common ones:")
    md("- Distance from " + tex_inline(r"(x,y,z)") + " to the **xy-plane** " + tex_inline(r"z=0") + " is " + tex_inline(r"|z|") + ".")
    md("- Distance to the **x-axis** is the distance to the line where " + tex_inline(r"y=0, z=0") + ":")
    tex_block(r"\sqrt{y^2+z^2}")
    md("**Goal:** do these instantly.")

def L4_mission():
    md("### Mission")
    md("For " + tex_inline(r"P=(2,-2,7)") + ", compute:")
    md("1) distance to the xy-plane")
    md("2) distance to the x-axis")

def L4_ui():
    P_in = st.text_input("Point P =", value="[2,-2,7]", key="L4_P")
    dxy = st.text_input("Distance to xy-plane =", value="", key="L4_dxy")
    dx = st.text_input("Distance to x-axis =", value="", key="L4_dx")
    return {"P_in": P_in, "dxy": dxy, "dx": dx}

def L4_check(data):
    try:
        P = parse_vec(data["P_in"])
        target_xy = sp.Abs(P[2])
        target_x = sp.sqrt(P[1]**2 + P[2]**2)
        user_xy = safe_sympify(data["dxy"])
        user_x = safe_sympify(data["dx"])
        ok1 = sp.simplify(user_xy - target_xy) == 0
        ok2 = sp.simplify(user_x - target_x) == 0
        if ok1 and ok2:
            return True, f"Correct: to xy-plane = {target_xy}, to x-axis = {sp.simplify(target_x)}."
        msg = []
        if not ok1:
            msg.append("xy-plane: use |z|.")
        if not ok2:
            msg.append("x-axis: use sqrt(y^2+z^2).")
        return False, " ".join(msg)
    except Exception as e:
        return False, f"Error: {e}"

def L4_viz(data):
    try:
        P = parse_vec(data["P_in"])
        Pnp = np.array([float(sp.N(P[i])) for i in range(3)], dtype=float)
        # show point + axes
        fig = go.Figure()
        fig.add_trace(go.Scatter3d(x=[Pnp[0]], y=[Pnp[1]], z=[Pnp[2]], mode="markers+text", text=["P"], textposition="top center", name="P"))
        axis_len = max(6, int(np.linalg.norm(Pnp) + 2))
        fig.add_trace(go.Scatter3d(x=[-axis_len, axis_len], y=[0,0], z=[0,0], mode="lines", name="x-axis", opacity=0.35))
        fig.add_trace(go.Scatter3d(x=[0,0], y=[-axis_len, axis_len], z=[0,0], mode="lines", name="y-axis", opacity=0.35))
        fig.add_trace(go.Scatter3d(x=[0,0], y=[0,0], z=[-axis_len, axis_len], mode="lines", name="z-axis", opacity=0.35))
        # projection to xy-plane
        fig.add_trace(go.Scatter3d(x=[Pnp[0], Pnp[0]], y=[Pnp[1], Pnp[1]], z=[Pnp[2], 0], mode="lines", name="to xy-plane", opacity=0.8))
        fig.update_layout(
            title="Point P and distance to the xy-plane",
            scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z", aspectmode="cube"),
            margin=dict(l=0, r=0, t=40, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        pass

# -----------------------------
# Level 5: Vector derivative + speed
# -----------------------------
def L5_teach():
    md("### Vector functions: derivative and speed")
    md("A vector function is")
    tex_block(r"\vec r(t)=\langle x(t), y(t), z(t)\rangle")
    md("Differentiate componentwise:")
    tex_block(r"\vec r'(t)=\langle x'(t), y'(t), z'(t)\rangle")
    md("Speed is the magnitude:")
    tex_block(r"|\vec r'(t)|=\sqrt{(x')^2+(y')^2+(z')^2}")
    md("**Goal:** compute speed at a time.")

def L5_mission():
    md("### Mission")
    md("For " + tex_inline(r"\vec r(t)=\langle t, 3\cos t, 3\sin t\rangle")
       + ", find the speed at " + tex_inline(r"t=0") + ".")

def L5_ui():
    r_in = st.text_input("r(t) =", value="[t, 3*cos(t), 3*sin(t)]", key="L5_r")
    t0_in = st.text_input("t0 =", value="0", key="L5_t0")
    ans = st.text_input("Speed |r'(t0)| =", value="", key="L5_ans")
    return {"r_in": r_in, "t0": t0_in, "ans": ans}

def L5_check(data):
    try:
        r = parse_vec(data["r_in"])
        rp = sp.diff(r, t)
        speed = sp.simplify(sp.sqrt(rp.dot(rp)))
        t0 = safe_sympify(data["t0"])
        target = sp.simplify(speed.subs(t, t0))
        user = safe_sympify(data["ans"])
        if sp.simplify(user - target) == 0:
            return True, f"Correct: r'(t) = {rp.T}, speed at t0 is {target}."
        return False, "Not quite. Differentiate each component, then take sqrt of sum of squares."
    except Exception as e:
        return False, f"Error: {e}"

def L5_viz(data):
    plot_curve_r_t()

# -----------------------------
# Level 6: Arc length
# -----------------------------
def L6_teach():
    md("### Arc length of a curve")
    md("Arc length of " + tex_inline(r"\vec r(t)") + " from " + tex_inline(r"t=a") + " to " + tex_inline(r"t=b") + " is:")
    tex_block(r"L=\int_a^b |\vec r'(t)|\,dt")
    md("Algorithm:")
    md("1) compute " + tex_inline(r"\vec r'(t)") )
    md("2) compute speed " + tex_inline(r"|\vec r'(t)|"))
    md("3) integrate")
    md("**Goal:** compute arc length.")

def L6_mission():
    md("### Mission")
    md("Find the arc length of " + tex_inline(r"\vec r(t)=\langle t,3\cos t,3\sin t\rangle")
       + " for " + tex_inline(r"-5\le t\le 5") + ".")

def L6_ui():
    r_in = st.text_input("r(t) =", value="[t, 3*cos(t), 3*sin(t)]", key="L6_r")
    a_in = st.text_input("a =", value="-5", key="L6_a")
    b_in = st.text_input("b =", value="5", key="L6_b")
    ans = st.text_input("Arc length L =", value="", key="L6_ans")
    return {"r_in": r_in, "a": a_in, "b": b_in, "ans": ans}

def L6_check(data):
    try:
        r = parse_vec(data["r_in"])
        rp = sp.diff(r, t)
        speed = sp.simplify(sp.sqrt(rp.dot(rp)))
        a = safe_sympify(data["a"])
        b = safe_sympify(data["b"])
        target = sp.simplify(sp.integrate(speed, (t, a, b)))
        user = safe_sympify(data["ans"])
        if sp.simplify(user - target) == 0:
            return True, f"Correct: |r'(t)| = {speed}, so L = {target}."
        return False, "Not quite. Use L = ∫ |r'(t)| dt. Here |r'(t)| simplifies to a constant."
    except Exception as e:
        return False, f"Error: {e}"

def L6_viz(data):
    plot_curve_r_t()
    md("Notice the curve is a helix: constant radius 3 around the x-axis, moving forward in x.")

# -----------------------------
# Level 7: Reparam by arc length (concept)
# -----------------------------
def L7_teach():
    md("### Reparametrization by arc length")
    md("Define arc length from a start time " + tex_inline(r"t_0") + ":")
    tex_block(r"s(t)=\int_{t_0}^{t}|\vec r'(u)|\,du")
    md("If speed is constant " + tex_inline(r"|\vec r'(t)|=v") + ", then:")
    tex_block(r"s=v(t-t_0)")
    md("**Goal:** build s(t) and invert if needed.")

def L7_mission():
    md("### Mission")
    md("For the helix " + tex_inline(r"\vec r(t)=\langle t,3\cos t,3\sin t\rangle")
       + ", the speed is constant. If " + tex_inline(r"t_0=0") + ", fill in:")
    tex_block(r"s(t)=\_\_\_\_")

def L7_ui():
    ans = st.text_input("Your s(t) =", value="", key="L7_ans")
    return {"ans": ans}

def L7_check(data):
    try:
        # speed is sqrt(1^2 + (-3 sin t)^2 + (3 cos t)^2) = sqrt(1 + 9) = sqrt(10)
        user = safe_sympify(data["ans"])
        target = sp.sqrt(10) * t
        if sp.simplify(user - target) == 0:
            return True, "Correct: s(t)=√10·t when starting at t0=0."
        return False, "Hint: speed is √10, so s=∫0^t √10 du = √10 t."
    except Exception as e:
        return False, f"Error: {e}"

def L7_viz(data):
    plot_curve_r_t()
    tex_block(r"|\vec r'(t)|=\sqrt{10}\ \Rightarrow\ s(t)=\int_0^t \sqrt{10}\,du=\sqrt{10}\,t")

# -----------------------------
# Level 8: Traces / level curves
# -----------------------------
def L8_teach():
    md("### Traces and level curves")
    md("For a surface " + tex_inline(r"z=f(x,y)") + ":")
    md("- Fix " + tex_inline(r"z=c") + " → level curve " + tex_inline(r"f(x,y)=c") + " in the xy-plane.")
    md("- Fix " + tex_inline(r"x=a") + " or " + tex_inline(r"y=b") + " → vertical slices.")
    md("Example: " + tex_inline(r"z=x^2+y^2") + ". If " + tex_inline(r"z=1") + ", then " + tex_inline(r"x^2+y^2=1") + " (a circle).")
    md("**Goal:** recognize traces fast.")

def L8_mission():
    md("### Mission")
    md("For the surface " + tex_inline(r"z=x^2+y^2") + ", what is the trace when " + tex_inline(r"z=1") + "?")

def L8_ui():
    choice = st.radio("Choose one:", ["a point", "a line", "a circle", "a parabola"], key="L8_choice")
    return {"choice": choice}

def L8_check(data):
    if data["choice"] == "a circle":
        return True, "Correct: z=1 → x^2 + y^2 = 1, a circle."
    return False, "Hint: set z=1 to get x^2 + y^2 = 1."

def L8_viz(data):
    plot_trace_circle_z_equals_1()

# -----------------------------
# Build levels list
# -----------------------------
LEVELS = [
    Level("L1", "Vector Basics: Dot Product", L1_teach, L1_mission, L1_ui, L1_check, L1_viz),
    Level("L2", "Cross Product: Perpendicular + Area", L2_teach, L2_mission, L2_ui, L2_check, L2_viz),
    Level("L3", "Planes from a Normal Vector", L3_teach, L3_mission, L3_ui, L3_check, L3_viz),
    Level("L4", "Distances in 3D", L4_teach, L4_mission, L4_ui, L4_check, L4_viz),
    Level("L5", "Vector Functions: Derivative + Speed", L5_teach, L5_mission, L5_ui, L5_check, L5_viz),
    Level("L6", "Arc Length", L6_teach, L6_mission, L6_ui, L6_check, L6_viz),
    Level("L7", "Reparametrize by Arc Length", L7_teach, L7_mission, L7_ui, L7_check, L7_viz),
    Level("L8", "Traces / Level Curves", L8_teach, L8_mission, L8_ui, L8_check, L8_viz),
]

# -----------------------------
# Header / HUD
# -----------------------------
st.title("🧭 Calc 3 Quest")
st.caption("Learn Calc 3 from scratch by playing. Read → Try → Get feedback → Visualize → Level up.")

# Sidebar controls
with st.sidebar:
    st.header("Game Controls")
    st.session_state.practice_mode = st.toggle("Practice mode (no lives lost)", value=st.session_state.practice_mode)
    if st.button("Reset game"):
        st.session_state.level_idx = 0
        st.session_state.xp = 0
        st.session_state.lives = 3
        st.session_state.attempts = 0
        st.session_state.completed = set()
        st.rerun()

# Progress bar
progress = (st.session_state.level_idx) / (len(LEVELS))
st.progress(progress)

# HUD metrics
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Level", f"{st.session_state.level_idx+1}/{len(LEVELS)}")
with c2:
    st.metric("XP", st.session_state.xp)
with c3:
    st.metric("Lives", "❤️" * st.session_state.lives)

hr()

# -----------------------------
# "Map" view
# -----------------------------
md("### 🗺️ Map")
map_cols = st.columns(4)
for i, lvl in enumerate(LEVELS):
    col = map_cols[i % 4]
    with col:
        done = lvl.id in st.session_state.completed
        label = f"{'✅' if done else '⬜'} {lvl.id}"
        st.button(label, disabled=True, use_container_width=True)

hr()

# -----------------------------
# Current level
# -----------------------------
current = LEVELS[st.session_state.level_idx]

md(f"## {current.title}")

with st.expander("Lesson (read this first)", expanded=True):
    current.teach_render()

hr()

current.mission_render()
inputs = current.ui()

# Visualization section
with st.expander("Visualize (highly recommended)", expanded=True):
    if current.viz:
        current.viz(inputs)
    else:
        md("No visualization for this level.")

hr()

# Buttons
colA, colB, colC = st.columns(3)
with colA:
    submit = st.button("Submit ✅", use_container_width=True)
with colB:
    hint_btn = st.button("Hint 💡", use_container_width=True)
with colC:
    skip = st.button("Skip (debug) ➡️", use_container_width=True)

# Hint logic
if hint_btn:
    hints = {
        "L1": "Dot product: multiply matching components and add.",
        "L2": "Cross product: determinant method. Remember the middle component gets a minus sign.",
        "L3": "Compute d by plugging the given point into ax+by+cz.",
        "L4": "xy-plane distance is |z|. x-axis distance is sqrt(y^2+z^2).",
        "L5": "Differentiate each component. Speed = sqrt((x')^2+(y')^2+(z')^2).",
        "L6": "Arc length = ∫|r'(t)|dt. For the helix, speed is constant.",
        "L7": "If speed is constant v, then s(t)=v t when starting at t0=0.",
        "L8": "Set z=1 to get x^2+y^2=1.",
    }
    hint(hints.get(current.id, "Break it into steps, and check your algebra."))

# Skip
if skip:
    if st.session_state.level_idx < len(LEVELS) - 1:
        st.session_state.level_idx += 1
        reset_attempts()
        st.rerun()

# Submit logic
if submit:
    st.session_state.attempts += 1
    ok, feedback = current.check(inputs)

    if ok:
        # XP: reward fewer tries
        gained = max(6, 18 - 3 * (st.session_state.attempts - 1))
        add_xp(gained)
        st.session_state.completed.add(current.id)
        reset_attempts()
        badge(f"+{gained} XP — {feedback}")

        if st.session_state.level_idx < len(LEVELS) - 1:
            if st.button("Next level ➡️", use_container_width=True):
                st.session_state.level_idx += 1
                st.rerun()
        else:
            st.balloons()
            badge(f"You finished Calc 3 Quest with {st.session_state.xp} XP! 🎉")
            md("### You can now:")
            md("- Compute dot/cross products and interpret geometry.")
            md("- Build plane equations from normals + points.")
            md("- Compute distances to planes/axes.")
            md("- Differentiate vector functions and compute speed.")
            md("- Compute arc length and understand arc-length parametrization.")
            md("- Use traces/level curves to understand surfaces.")
    else:
        if st.session_state.practice_mode:
            note("Practice mode is ON — you won't lose lives.")
            fail(feedback)
        else:
            st.session_state.lives -= 1
            fail(feedback)
            if st.session_state.lives <= 0:
                st.error("Game Over — out of lives.")
                md("Turn on **Practice mode** in the sidebar, or hit reset.")
            else:
                note(f"You lost a life. Lives left: {'❤️'*st.session_state.lives}")

hr()

with st.expander("Extra Practice Generator (optional)", expanded=False):
    md("This is optional replay practice. It doesn’t affect your progression.")
    topic = st.selectbox("Pick a topic", ["Dot product", "Distance to xy-plane", "Speed of r(t)=<t,3cos t,3sin t>"])
    if st.button("Generate"):
        if topic == "Dot product":
            a = np.random.randint(-4, 5, size=3)
            b = np.random.randint(-4, 5, size=3)
            md(f"Compute dot product of v={a.tolist()} and w={b.tolist()}.")
            md(f"(Answer: {int(np.dot(a,b))})")
            plot_vectors_3d(a.astype(float), b.astype(float), title="Practice vectors")
        elif topic == "Distance to xy-plane":
            P = np.random.randint(-9, 10, size=3)
            md(f"Point P={P.tolist()}. Distance to xy-plane?")
            md(f"(Answer: {abs(int(P[2]))})")
        else:
            md("For r(t)=<t,3cos t,3sin t>, speed is constant.")
            md("(Answer: √10)")

st.caption("If you want, I can add: gradients/tangent planes, directional derivatives, double/triple integrals, and full boss fights.")
