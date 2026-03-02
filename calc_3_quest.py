import streamlit as st
import numpy as np
import sympy as sp
from dataclasses import dataclass
from typing import Callable, Dict, Any, Tuple, List

# -----------------------------
# Helpers
# -----------------------------
x, y, z, t = sp.symbols('x y z t', real=True)

def almost_equal(a, b, tol=1e-6):
    try:
        return abs(float(a) - float(b)) <= tol
    except Exception:
        return False

def vec3(a, b, c):
    return sp.Matrix([a, b, c])

def safe_sympify(expr: str):
    # allow common functions
    locals_dict = {
        "pi": sp.pi,
        "sqrt": sp.sqrt,
        "sin": sp.sin,
        "cos": sp.cos,
        "tan": sp.tan,
        "exp": sp.exp,
        "ln": sp.log,
        "log": sp.log,
        "abs": sp.Abs
    }
    return sp.sympify(expr, locals=locals_dict)

def parse_vec(expr: str):
    """
    Parse a vector from inputs like:
    [1,2,3] or <1,2,3> or (1,2,3)
    """
    s = expr.strip().replace("<", "[").replace(">", "]").replace("(", "[").replace(")", "]")
    v = safe_sympify(s)
    if not (isinstance(v, (list, tuple, sp.Matrix, sp.ImmutableMatrix))):
        # sometimes sympify returns Matrix for [a,b,c]
        pass
    if isinstance(v, (list, tuple)):
        v = sp.Matrix(v)
    if isinstance(v, (sp.Matrix, sp.ImmutableMatrix)) and v.shape == (3, 1):
        return v
    if isinstance(v, (sp.Matrix, sp.ImmutableMatrix)) and v.shape == (1, 3):
        return v.T
    raise ValueError("Could not parse a 3D vector. Try format like [1,2,3].")

def format_vec(v: sp.Matrix):
    return f"<{sp.simplify(v[0])}, {sp.simplify(v[1])}, {sp.simplify(v[2])}>"

def distance_point_to_plane(P: sp.Matrix, n: sp.Matrix, d: sp.Expr):
    # plane: n·<x,y,z> = d
    # dist = |n·P - d| / ||n||
    return sp.Abs(n.dot(P) - d) / sp.sqrt(n.dot(n))

def distance_point_to_axis(P: sp.Matrix, axis: str):
    # distance from (x,y,z) to x-axis => sqrt(y^2+z^2), etc.
    if axis == "x":
        return sp.sqrt(P[1]**2 + P[2]**2)
    if axis == "y":
        return sp.sqrt(P[0]**2 + P[2]**2)
    if axis == "z":
        return sp.sqrt(P[0]**2 + P[1]**2)
    raise ValueError("axis must be x,y,z")

def domain_notes(expr: sp.Expr) -> str:
    # Very lightweight domain hints: warns about division by 0, sqrt of negative (real), log arg <=0
    notes = []
    # division by 0
    denoms = sp.denom(expr)
    if denoms != 1:
        notes.append(f"Avoid values where the denominator = 0: {sp.factor(denoms)} ≠ 0.")
    # sqrt
    for r in sp.preorder_traversal(expr):
        if isinstance(r, sp.Pow) and r.exp == sp.Rational(1, 2):
            notes.append(f"For real outputs, require inside sqrt ≥ 0: {r.base} ≥ 0.")
    # log
    for r in sp.preorder_traversal(expr):
        if r.func == sp.log:
            notes.append(f"For real outputs, require log argument > 0: {r.args[0]} > 0.")
    if not notes:
        notes.append("No obvious restrictions found (over the reals), but always check carefully.")
    return "\n".join(f"• {n}" for n in notes)

# -----------------------------
# Game Structure
# -----------------------------
@dataclass
class Level:
    id: str
    title: str
    teach: str
    problem: str
    answer_check: Callable[[Dict[str, Any]], Tuple[bool, str]]  # returns (correct, feedback)
    ui: Callable[[], Dict[str, Any]]  # returns user inputs

def badge(msg):
    st.markdown(f"✅ **{msg}**")

def fail(msg):
    st.markdown(f"❌ **{msg}**")

def hint(msg):
    st.info(msg)

def math_block(s):
    st.latex(s)

def section_header(title):
    st.markdown(f"## {title}")

# -----------------------------
# Levels (teach + try)
# -----------------------------
def level_1_ui():
    st.write("Enter a vector **v** and a vector **w** as `[a,b,c]`.")
    v_in = st.text_input("v =", value="[1,2,3]")
    w_in = st.text_input("w =", value="[4,0,-1]")
    return {"v_in": v_in, "w_in": w_in}

def level_1_check(data):
    try:
        v = parse_vec(data["v_in"])
        w = parse_vec(data["w_in"])
        dot = sp.simplify(v.dot(w))
        if dot == sp.Integer(1*4 + 2*0 + 3*(-1)):
            return True, f"Nice! v·w = {dot}."
        else:
            # even if their v,w differ from defaults, accept if they computed dot correctly? Here we are teaching;
            # We'll check they computed dot for the specific v,w they typed by asking them to compute it.
            return False, "For this level, use the default v and w values (or reset them) so we’re synced."
    except Exception as e:
        return False, f"Parse error: {e}"

level_1 = Level(
    id="L1",
    title="Vector Basics: Dot Product",
    teach=r"""
You’ll use vectors constantly in Calc 3.

**Dot product** measures how aligned two vectors are.

If \(\vec v=\langle v_1,v_2,v_3\rangle\) and \(\vec w=\langle w_1,w_2,w_3\rangle\), then  
\[
\vec v\cdot \vec w = v_1w_1+v_2w_2+v_3w_3.
\]

If \(\vec v\cdot \vec w=0\), they’re perpendicular.

**Goal:** compute a dot product accurately.
""",
    problem=r"Compute \(\vec v\cdot \vec w\) for \(\vec v=\langle 1,2,3\rangle\) and \(\vec w=\langle 4,0,-1\rangle\).",
    ui=level_1_ui,
    answer_check=level_1_check
)

def level_2_ui():
    st.write("Compute a cross product and interpret it.")
    v_in = st.text_input("v =", value="[1,2,0]")
    w_in = st.text_input("w =", value="[0,1,3]")
    ans = st.text_input("Your v×w =", value="[ ]")
    return {"v_in": v_in, "w_in": w_in, "ans": ans}

def level_2_check(data):
    try:
        v = parse_vec(data["v_in"])
        w = parse_vec(data["w_in"])
        target = sp.simplify(v.cross(w))
        user = parse_vec(data["ans"])
        if all(sp.simplify(user[i] - target[i]) == 0 for i in range(3)):
            area = sp.sqrt(target.dot(target))
            return True, f"Correct! v×w = {format_vec(target)}. Its magnitude is the parallelogram area = {sp.simplify(area)}."
        return False, f"Not quite. Hint: use the determinant formula. Target is {format_vec(target)}."
    except Exception as e:
        return False, f"Parse error: {e}"

level_2 = Level(
    id="L2",
    title="Cross Product: Perpendicular Vector + Area",
    teach=r"""
The **cross product** produces a vector perpendicular to both inputs.

\[
\vec v\times \vec w =
\begin{vmatrix}
\hat i & \hat j & \hat k \\
v_1 & v_2 & v_3 \\
w_1 & w_2 & w_3
\end{vmatrix}
\]

Key facts:
- Direction: perpendicular to both (right-hand rule).
- Magnitude: \(|\vec v\times\vec w|\) = area of the parallelogram spanned by \(v,w\).

**Goal:** compute \(v\times w\).
""",
    problem=r"Find \(\vec v\times\vec w\) for \(\vec v=\langle 1,2,0\rangle\) and \(\vec w=\langle 0,1,3\rangle\).",
    ui=level_2_ui,
    answer_check=level_2_check
)

def level_3_ui():
    st.write("Plane equation from a point and normal vector.")
    n_in = st.text_input("Normal n =", value="[2,-1,3]")
    P0_in = st.text_input("Point P0 =", value="[1,0,2]")
    ans = st.text_input("Plane equation in form ax+by+cz=d. Enter d =", value="")
    return {"n_in": n_in, "P0_in": P0_in, "d": ans}

def level_3_check(data):
    try:
        n = parse_vec(data["n_in"])
        P0 = parse_vec(data["P0_in"])
        d_target = sp.simplify(n.dot(P0))
        d_user = safe_sympify(data["d"])
        if sp.simplify(d_user - d_target) == 0:
            return True, f"Yes! Plane is {n[0]}x + {n[1]}y + {n[2]}z = {d_target}."
        return False, f"Close — compute d = n·P0. Here d should be {d_target}."
    except Exception as e:
        return False, f"Error: {e}"

level_3 = Level(
    id="L3",
    title="Planes from a Normal Vector",
    teach=r"""
A plane is easy if you know:
- a point \(P_0=(x_0,y_0,z_0)\)
- a normal vector \(\vec n=\langle a,b,c\rangle\)

Then every point \((x,y,z)\) on the plane satisfies:
\[
\vec n\cdot\langle x-x_0, y-y_0, z-z_0\rangle = 0
\]
which expands to:
\[
ax+by+cz = d,\quad \text{where } d=ax_0+by_0+cz_0.
\]

**Goal:** build the plane equation.
""",
    problem=r"For \(\vec n=\langle 2,-1,3\rangle\) and \(P_0=(1,0,2)\), find \(d\) in \(2x-y+3z=d\).",
    ui=level_3_ui,
    answer_check=level_3_check
)

def level_4_ui():
    st.write("Distance from a point to the xy-plane and x-axis.")
    P_in = st.text_input("Point P =", value="[2,-2,7]")
    ans_xy = st.text_input("Distance to xy-plane =", value="")
    ans_x = st.text_input("Distance to x-axis =", value="")
    return {"P_in": P_in, "ans_xy": ans_xy, "ans_x": ans_x}

def level_4_check(data):
    try:
        P = parse_vec(data["P_in"])
        d_xy_target = sp.Abs(P[2])
        d_x_target = sp.simplify(distance_point_to_axis(P, "x"))
        d_xy_user = safe_sympify(data["ans_xy"])
        d_x_user = safe_sympify(data["ans_x"])

        ok1 = sp.simplify(d_xy_user - d_xy_target) == 0
        ok2 = sp.simplify(d_x_user - d_x_target) == 0

        if ok1 and ok2:
            return True, f"Correct! To xy-plane: {d_xy_target}. To x-axis: {d_x_target}."
        msg = []
        if not ok1:
            msg.append("For xy-plane, distance is just |z|.")
        if not ok2:
            msg.append("For x-axis, distance is sqrt(y^2+z^2).")
        return False, " ".join(msg) + f" Targets: {d_xy_target} and {d_x_target}."
    except Exception as e:
        return False, f"Error: {e}"

level_4 = Level(
    id="L4",
    title="Distances in 3D",
    teach=r"""
Distance ideas you’ll reuse:

- Distance from \((x,y,z)\) to the **xy-plane** (\(z=0\)) is \(|z|\).
- Distance from \((x,y,z)\) to the **x-axis** is the distance to the line where \(y=0, z=0\):
\[
\sqrt{y^2+z^2}.
\]

**Goal:** compute distances quickly.
""",
    problem=r"For \(P=(2,-2,7)\), compute distance to the xy-plane and to the x-axis.",
    ui=level_4_ui,
    answer_check=level_4_check
)

def level_5_ui():
    st.write("Vector function derivative + speed.")
    r_in = st.text_input("Enter r(t) as [x(t),y(t),z(t)]", value="[t, 3*cos(t), 3*sin(t)]")
    t0_in = st.text_input("Evaluate speed at t0 =", value="0")
    ans = st.text_input("Speed |r'(t0)| =", value="")
    return {"r_in": r_in, "t0": t0_in, "ans": ans}

def level_5_check(data):
    try:
        r = parse_vec(data["r_in"])
        rp = sp.diff(r, t)
        speed = sp.simplify(sp.sqrt(rp.dot(rp)))
        t0 = safe_sympify(data["t0"])
        target = sp.simplify(speed.subs(t, t0))
        user = safe_sympify(data["ans"])
        if sp.simplify(user - target) == 0:
            return True, f"Yes! r'(t) = {format_vec(rp)} and speed = {target} at t={t0}."
        return False, f"Not quite. Compute r'(t) componentwise, then |r'(t)|. Target at t0 is {target}."
    except Exception as e:
        return False, f"Error: {e}"

level_5 = Level(
    id="L5",
    title="Vector Functions: Derivative and Speed",
    teach=r"""
A **vector function** is:
\[
\vec r(t) = \langle x(t), y(t), z(t)\rangle
\]

Derivative is componentwise:
\[
\vec r'(t) = \langle x'(t), y'(t), z'(t)\rangle
\]

Speed is magnitude:
\[
\text{speed} = |\vec r'(t)| = \sqrt{(x')^2+(y')^2+(z')^2}.
\]

**Goal:** compute speed at a specific time.
""",
    problem=r"For \(\vec r(t)=\langle t, 3\cos t, 3\sin t\rangle\), find the speed at \(t=0\).",
    ui=level_5_ui,
    answer_check=level_5_check
)

def level_6_ui():
    st.write("Arc length of a vector curve.")
    r_in = st.text_input("r(t) =", value="[t, 3*cos(t), 3*sin(t)]")
    a_in = st.text_input("a =", value="-5")
    b_in = st.text_input("b =", value="5")
    ans = st.text_input("Arc length =", value="")
    return {"r_in": r_in, "a": a_in, "b": b_in, "ans": ans}

def level_6_check(data):
    try:
        r = parse_vec(data["r_in"])
        rp = sp.diff(r, t)
        speed = sp.simplify(sp.sqrt(rp.dot(rp)))
        a = safe_sympify(data["a"])
        b = safe_sympify(data["b"])
        target = sp.simplify(sp.integrate(speed, (t, a, b)))
        user = safe_sympify(data["ans"])
        if sp.simplify(user - target) == 0:
            return True, f"Correct! |r'(t)| = {sp.simplify(speed)} so length = {target}."
        return False, f"Not quite. Length = ∫_a^b |r'(t)| dt. Here |r'(t)| simplifies nicely. Target: {target}."
    except Exception as e:
        return False, f"Error: {e}"

level_6 = Level(
    id="L6",
    title="Arc Length",
    teach=r"""
Arc length of a curve \(\vec r(t)\), \(t\in[a,b]\):
\[
L = \int_a^b |\vec r'(t)|\,dt.
\]

So the algorithm is:
1) Differentiate \(\vec r(t)\)  
2) Compute speed \(|\vec r'(t)|\)  
3) Integrate.

**Goal:** compute arc length.
""",
    problem=r"Find the arc length of \(\vec r(t)=\langle t, 3\cos t, 3\sin t\rangle\) from \(t=-5\) to \(t=5\).",
    ui=level_6_ui,
    answer_check=level_6_check
)

def level_7_ui():
    st.write("Reparametrize by arc length (conceptual + formula).")
    st.write("For the same curve r(t) = <t,3cos t,3sin t>, the speed is constant.")
    ans = st.text_input("If speed = v, then arc length s from 0 to t is s = ____", value="")
    return {"ans": ans}

def level_7_check(data):
    try:
        user = safe_sympify(data["ans"])
        # speed is sqrt(1+9) = sqrt(10)
        target = sp.sqrt(10)*t
        if sp.simplify(user - target) == 0:
            return True, "Yes! With constant speed v, s = v t (when starting at t=0)."
        return False, f"Hint: s(t)=∫0^t |r'(u)| du. Here speed is constant √10, so s=√10 t."
    except Exception as e:
        return False, f"Error: {e}"

level_7 = Level(
    id="L7",
    title="Reparametrization by Arc Length",
    teach=r"""
To reparametrize by arc length, define:
\[
s(t)=\int_{t_0}^t |\vec r'(u)|\,du
\]
Then solve for \(t\) as a function of \(s\), and substitute into \(\vec r(t)\).

When **speed is constant** \( |\vec r'(t)| = v\):
\[
s = v(t-t_0).
\]

**Goal:** build the arc-length parameter relationship.
""",
    problem=r"For \(t_0=0\) and speed \(v=\sqrt{10}\), fill in \(s=\_\_\_\_\).",
    ui=level_7_ui,
    answer_check=level_7_check
)

def level_8_ui():
    st.write("Traces / slices: identify what you get when you set a variable constant.")
    st.write("Surface: z = x^2 + y^2")
    choice = st.radio("What is the trace when z = 1?", ["a point", "a line", "a circle", "a parabola"])
    return {"choice": choice}

def level_8_check(data):
    if data["choice"] == "a circle":
        return True, "Correct: setting z=1 gives x^2 + y^2 = 1, which is a circle in the xy-plane."
    return False, "Hint: z=1 means x^2 + y^2 = 1."

level_8 = Level(
    id="L8",
    title="Traces and Level Curves",
    teach=r"""
A surface given by \(z=f(x,y)\) can be understood by **traces**:

- Fix \(z=c\) gives a **level curve**: \(f(x,y)=c\) in the \(xy\)-plane.
- Fix \(x=a\) or \(y=b\) gives vertical slices.

Example: \(z=x^2+y^2\).  
If \(z=1\), then \(x^2+y^2=1\) → a circle.

**Goal:** recognize traces quickly.
""",
    problem="For the surface z = x^2 + y^2, what is the trace when z = 1?",
    ui=level_8_ui,
    answer_check=level_8_check
)

LEVELS = [level_1, level_2, level_3, level_4, level_5, level_6, level_7, level_8]

# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(page_title="Calc 3 Quest", page_icon="🧭", layout="centered")

st.title("🧭 Calc 3 Quest")
st.caption("A mini RPG that teaches Calc 3 concepts from scratch by playing.")

if "level_idx" not in st.session_state:
    st.session_state.level_idx = 0
if "xp" not in st.session_state:
    st.session_state.xp = 0
if "lives" not in st.session_state:
    st.session_state.lives = 3
if "completed" not in st.session_state:
    st.session_state.completed = set()
if "attempts" not in st.session_state:
    st.session_state.attempts = 0

current = LEVELS[st.session_state.level_idx]

# HUD
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Level", f"{st.session_state.level_idx+1}/{len(LEVELS)}")
with col2:
    st.metric("XP", st.session_state.xp)
with col3:
    st.metric("Lives", "❤️"*st.session_state.lives)

st.divider()

section_header(current.title)
st.markdown(current.teach)
st.markdown("### Mission")
st.markdown(current.problem)

st.divider()

inputs = current.ui()

colA, colB = st.columns([1, 1])
with colA:
    submit = st.button("Submit Answer ✅", use_container_width=True)
with colB:
    get_hint = st.button("Hint 💡", use_container_width=True)

if get_hint:
    # Provide generic hint based on level id
    if current.id in ["L1"]:
        hint("Dot product: multiply matching components and add them.")
    elif current.id in ["L2"]:
        hint("Cross product: use determinant or component formula. Be careful with signs.")
    elif current.id in ["L3"]:
        hint("Compute d by plugging the given point into ax+by+cz.")
    elif current.id in ["L4"]:
        hint("xy-plane distance is |z|. x-axis distance uses only y and z.")
    elif current.id in ["L5"]:
        hint("Differentiate each component. Then speed is sqrt of sum of squares.")
    elif current.id in ["L6"]:
        hint("Arc length is integral of speed |r'(t)| from a to b.")
    elif current.id in ["L7"]:
        hint("If speed is constant v, then s = v*t (starting at 0).")
    elif current.id in ["L8"]:
        hint("Setting z=1 turns the surface equation into a relation between x and y.")
    else:
        hint("Try breaking the problem into steps, and check units/meaning.")

if submit:
    st.session_state.attempts += 1
    correct, feedback = current.answer_check(inputs)

    if correct:
        badge(feedback)
        # XP scaling: fewer attempts => more XP
        gained = max(5, 15 - 2*(st.session_state.attempts - 1))
        st.session_state.xp += gained
        st.session_state.completed.add(current.id)
        st.session_state.attempts = 0

        if st.session_state.level_idx < len(LEVELS) - 1:
            st.success(f"+{gained} XP. Door unlocked! Proceed to the next level.")
            if st.button("Next Level ➡️", use_container_width=True):
                st.session_state.level_idx += 1
                st.rerun()
        else:
            st.balloons()
            st.success(f"You finished Calc 3 Quest with {st.session_state.xp} XP! 🎉")
            st.markdown("### What you can do now")
            st.markdown("""
- Solve basic vector, plane, distance, vector-derivative, speed, arc-length, and trace problems.
- Replay levels with different inputs (modify the vectors/functions) to deepen intuition.
            """)
    else:
        st.session_state.lives -= 1
        fail(feedback)
        if st.session_state.lives <= 0:
            st.error("Game over. You ran out of lives. Reset and try again!")
            if st.button("Reset Game 🔄", use_container_width=True):
                st.session_state.level_idx = 0
                st.session_state.xp = 0
                st.session_state.lives = 3
                st.session_state.completed = set()
                st.session_state.attempts = 0
                st.rerun()
        else:
            st.warning(f"You lost a life. Lives left: {st.session_state.lives}")

st.divider()
with st.expander("Teacher Console (optional)"):
    st.write("Use this to generate fresh practice prompts or see domain hints for expressions.")
    expr_in = st.text_input("Expression (in t) for domain hinting", value="(t+1)/(t-2)")
    try:
        expr = safe_sympify(expr_in)
        st.write("Domain hints:")
        st.write(domain_notes(expr))
    except Exception as e:
        st.write(f"Could not parse: {e}")

st.caption("Tip: This game is meant to teach from zero—students should read the lesson text each level before attempting the mission.")
