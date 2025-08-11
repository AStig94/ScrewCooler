import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar

# ------------------- How to -------------------
# If using VSC (else just use the streamlit app:    https://screwcooler.streamlit.app/
# Make sure you are in the right folder. "cd .." goes one folder up, "cd+space+shift" autofills destination.
# in terminal write:    streamlit run InteractiveApp_ScrewCooler.py
# ctrl+c to abort terminal..

# ------------------- Setup -------------------
st.set_page_config(page_title="Biochar Screw Cooler",layout="wide")
st.title("Biochar Screw Cooler")
st.markdown("Adjust the parameters to simulate the cooling screw behavior.")

# ------------------- Control Panel -------------------

# Simulation properties
st.sidebar.markdown("### Simulation Parameters")
t_biochar_in = st.sidebar.number_input("Biochar Initial Temperature (°C)", min_value=50, max_value=500, value=350, step=10)
T_target = st.sidebar.number_input("Biochar Target Temperature (°C)", min_value=30, max_value=350, value=30, step=10)
cool_shaft = st.sidebar.checkbox("Enable Shaft Cooling", True)

# Biochar flow rate (kg/hr)
m_biochar_kg_hr = st.sidebar.number_input("Biochar Mass Flow Rate (kg/hr)", min_value=10, max_value=300, value=80, step=10)
m_biochar = m_biochar_kg_hr / 3600  # convert to kg/s


# Screw parameters
st.sidebar.markdown("### Screw Parameters")
rpm = st.sidebar.slider("Screw RPM", 1, 8, 6)
screw_diameter = st.sidebar.slider("Screw Diameter (mm)", 100, 300, 140)
screw_pitch_ratio = st.sidebar.slider("Pitch / Diameter Ratio", 0.1, 1.5, 0.5)


# Biochar properties inputs
st.sidebar.markdown("### Biochar Properties")
rho_biochar = st.sidebar.number_input("Density (kg/m³)", min_value=100.0, max_value=2000.0, value=615.0, step=10.0)
C_biochar = st.sidebar.number_input("Heat Capacity (J/kg·K)", min_value=100.0, max_value=3000.0, value=1200.0, step=10.0)
lambda_biochar = st.sidebar.number_input("Thermal Conductivity (W/m·K)", min_value=0.01, max_value=1.0, value=0.12, step=0.01)

# Cooling duty calculation (kW) based on m*cp*ΔT
cooling_duty_kw = m_biochar * C_biochar * (t_biochar_in - T_target) / 1000.0

st.sidebar.markdown(
    f"""
    <div style="
        padding: 0.6em;
        margin-top: 0.5em;
        border-radius: 8px;
        background-color: #f0f2f6;
        border: 1px solid #d3d3d3;
        text-align: center;
        font-size: 1.2em;
        font-weight: bold;
        color: #333;">
        💧 Cooling Duty: <span style="color:#007acc;">{cooling_duty_kw:.2f} kW</span>
    </div>
    """,
    unsafe_allow_html=True
)

# ------------------- Constants -------------------
g = 9.81
pi = np.pi

# Water properties
rho_water = 997
C_water = 4180
mu_water = 0.00089
lambda_water = 0.606

# Steel
delta_steel = 0.002
lambda_steel = 16.3

# ------------------- Geometry -------------------
# Set base diameter and scale others proportionally
d3 = screw_diameter / 1000  # Convert mm to meters
r3 = d3 / 2

# Original ratios from your script
r1_ratio = 0.126 / 0.248
r5_ratio = 0.266 / 0.248

r1 = r3 * r1_ratio
r2 = r1 + delta_steel
r4 = r3 + delta_steel
r5 = r3 * r5_ratio
r6 = r5 + delta_steel

pitch = screw_pitch_ratio * (2 * r3)
n_rpm = rpm
n_rps = n_rpm / 60

# ------------------- Compute Biochar volumetric flow rate -------------------
q_v_biochar = m_biochar / rho_biochar

# ------------------- Calculate water flow rates based on velocity = 1 m/s -------------------
v_w = 1.0  # desired velocity m/s

# Shaft cooling water velocity and volumetric flow rate
D_h_s = r2 * 2
area_s = pi * r2**2
q_v_water_s = area_s * v_w

# Outer cooling water velocity and volumetric flow rate
D_h_c = (2 * r5) - (2 * r4)
area_c = pi * (r5**2 - r4**2)
q_v_water_c = area_c * v_w

# ------------------- Inlet temperatures -------------------
t_water_in = 20

# ------------------- Numerical settings -------------------
dx = 0.05
max_length = 10.0
steps = int(max_length / dx)
x_grid = np.linspace(0, max_length, steps)

# ------------------- Thermodynamic Functions -------------------
def alpha_biochar(lambda_biochar, rho_biochar, C_biochar, n, x_val, c_val, D):
    T = c_val * ((2 * pi * n)**2 * D / (2 * g))**x_val / n
    return 2 * np.sqrt(lambda_biochar * rho_biochar * C_biochar / (pi * T))

def alpha_water(rho, mu, cp, k, D, v):
    Re = rho * v * D / mu
    Pr = cp * mu / k
    if Re <= 2300:
        return 3.66 * k / D
    f = (0.79 * np.log(Re) - 1.64)**-2
    Nu = (f / 8) * (Re - 1000) * Pr / (1 + 12.7 * np.sqrt(f / 8) * (Pr**(2/3) - 1))
    return Nu * k / D

# ------------------- Geometry Functions -------------------
def wetted_arc_length(h, R):
    if h <= 0: return 0.0
    if h >= 2 * R: return 2 * np.pi * R
    return 2 * R * np.arccos(np.clip(1 - h / R, -1.0, 1.0))

def circular_segment_area(h, R):
    if h <= 0: return 0.0
    if h >= 2 * R: return np.pi * R**2
    theta = 2 * np.arccos(np.clip(1 - h / R, -1.0, 1.0))
    return 0.5 * R**2 * (theta - np.sin(theta))

def compute_wetted_perimeters(f_fill, r_w, r_s):
    A_annulus = np.pi * (r_w**2 - r_s**2)
    A_filled = f_fill * A_annulus

    def annulus_segment_area(h):
        A_outer = circular_segment_area(min(h, 2 * r_w), r_w)
        h_inner = max(h - (r_w - r_s), 0)
        A_inner = circular_segment_area(min(h_inner, 2 * r_s), r_s)
        return A_outer - A_inner

    def objective(h): return annulus_segment_area(h) - A_filled

    sol = root_scalar(objective, bracket=[0, 2 * r_w], method='brentq')
    h_char = sol.root if sol.converged else np.nan

    O_w = wetted_arc_length(h_char, r_w)
    h_s = max(h_char - (r_w - r_s), 0)
    O_s = wetted_arc_length(h_s, r_s)

    return O_w, O_s, h_char

def compute_geometry_and_wetted_perimeters(r2, r3, q_v_biochar, pitch, n_rps):
    A_annulus = np.pi * (r3**2 - r2**2)
    v_char = pitch * n_rps
    f_fill = q_v_biochar / (A_annulus * v_char)
    if f_fill > 1.0:
        st.error(f"❌ Degree of fill too high: {f_fill:.3f}. Please adjust parameters.")
        st.stop()
    O_w, O_s, h_char = compute_wetted_perimeters(f_fill, r3, r2)
    return f_fill, O_w, O_s, h_char

# ------------------- Calculations -------------------
f_fill, O_w, O_s, h_char = compute_geometry_and_wetted_perimeters(r2, r3, q_v_biochar, pitch, n_rps)

v_s_calc = q_v_water_s / (pi * r2**2)
v_c_calc = q_v_water_c / (pi * (r5**2 - r4**2))

if cool_shaft:
    alpha_steel_s = lambda_steel / ((r2) * np.log((r2) / r1))
    alpha_water_s = alpha_water(rho_water, mu_water, C_water, lambda_water, D_h_s, v_s_calc)
else:
    alpha_steel_s = np.inf
    alpha_water_s = np.inf

alpha_steel_c = lambda_steel / ((r4) * np.log((r4) / r3))
alpha_water_c = alpha_water(rho_water, mu_water, C_water, lambda_water, D_h_c, v_c_calc)

x_vals = [0.1, 0.3, 0.5, 0.7]
c = 4.0

# Create two columns for side by side plots
col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots(figsize=(6, 4), dpi=80)
    results = []
    cooling_results = []
    for x_model in x_vals:
        alpha_biochar_s = alpha_biochar(lambda_biochar, rho_biochar, C_biochar, n_rps, x_model, c, 2 * r2)
        alpha_biochar_c = alpha_biochar(lambda_biochar, rho_biochar, C_biochar, n_rps, x_model, c, 2 * r3)

        R_total_s = 1 / alpha_biochar_s + 1 / alpha_steel_s + 1 / alpha_water_s
        R_total_c = 1 / alpha_biochar_c + 1 / alpha_steel_c + 1 / alpha_water_c

        alpha_s = 1 / R_total_s
        alpha_c = 1 / R_total_c

        t_a = np.zeros(steps)
        t_s = np.zeros(steps)
        t_c = np.zeros(steps)
        t_a[0] = t_biochar_in
        t_s[0] = t_water_in
        t_c[0] = t_water_in

        for i in range(steps - 1):
            dQ_s = alpha_s * O_s * dx * (t_a[i] - t_s[i]) if cool_shaft else 0.0
            dQ_c = alpha_c * O_w * dx * (t_a[i] - t_c[i])

            t_a[i+1] = t_a[i] - (dQ_s + dQ_c) / (rho_biochar * C_biochar * q_v_biochar)
            t_s[i+1] = t_s[i] + dQ_s / (rho_water * C_water * q_v_water_s) if cool_shaft else t_s[i]
            t_c[i+1] = t_c[i] + dQ_c / (rho_water * C_water * q_v_water_c)

            if t_a[i+1] < T_target:
                required_length = x_grid[i+1]
                break
        else:
            required_length = max_length

        if required_length >= max_length:
            label = f"x = {x_model}, L > {max_length:.1f} m"
        else:
            label = f"x = {x_model}, L = {required_length:.2f} m"
        ax.plot(x_grid[:i+2], t_a[:i+2], label=label)
        results.append((x_model, required_length, alpha_s if cool_shaft else 0.0, alpha_c))

        t_s_out = t_s[i+1] if cool_shaft else None
        t_c_out = t_c[i+1]
        cooling_results.append({
        "x": x_model,
        "T_water_shaft_out (°C)": f"{t_s_out:.1f}" if t_s_out is not None else "N/A",
        "T_water_casing_out (°C)": f"{t_c_out:.1f}",
})

    ax.set_xlabel("Length along cooler (m)")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title("Biochar Cooling Profile")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig, use_container_width=True)

with col2:
    def plot_perimeters():
        r_s = r2
        r_w = r3
        O_w, O_s, h_char = compute_wetted_perimeters(f_fill, r_w, r_s)
        h_s = max(h_char - (r_w - r_s), 0)
        theta = np.linspace(0, 2*np.pi, 500)
        x_w = r_w * np.cos(theta)
        y_w = r_w * np.sin(theta)
        x_s = r_s * np.cos(theta)
        y_s = r_s * np.sin(theta)

        theta_outer_max = np.arccos(np.clip(1 - h_char / r_w, -1.0, 1.0))
        theta_inner_max = np.arccos(np.clip(1 - h_s / r_s, -1.0, 1.0))

        theta_wetted_outer = np.linspace(-theta_outer_max, theta_outer_max, 300) if h_char < 2*r_w else np.linspace(-np.pi, np.pi, 300)
        theta_wetted_inner = np.linspace(-theta_inner_max, theta_inner_max, 300) if h_s < 2*r_s else np.linspace(-np.pi, np.pi, 300)

        y_wo = r_w * np.cos(theta_wetted_outer)
        z_wo = r_w * np.sin(theta_wetted_outer)
        y_so = r_s * np.cos(theta_wetted_inner)
        z_so = r_s * np.sin(theta_wetted_inner)

        theta_inner_fill = theta_wetted_inner[::-1] if h_s < 2 * r_s else np.linspace(np.pi, -np.pi, 300)
        y_outer = r_w * np.sin(theta_wetted_outer)
        z_outer = -r_w * np.cos(theta_wetted_outer)
        y_inner = r_s * np.sin(theta_inner_fill)
        z_inner = -r_s * np.cos(theta_inner_fill)

        z_start = z_outer[0]
        y_start_outer = y_outer[0]
        y_start_inner = y_inner[0]

        z_end = z_outer[-1]
        y_end_outer = y_outer[-1]
        y_end_inner = y_inner[-1]

        y_connect_start = np.linspace(y_start_outer, y_start_inner, 2)
        z_connect_start = np.full_like(y_connect_start, z_start)
        y_connect_end = np.linspace(y_end_inner, y_end_outer, 2)
        z_connect_end = np.full_like(y_connect_end, z_end)

        y_fill = np.concatenate([y_outer, y_connect_start, y_inner, y_connect_end])
        z_fill = np.concatenate([z_outer, z_connect_start, z_inner, z_connect_end])

        fig, ax = plt.subplots(figsize=(6, 6), dpi=80)
        ax.plot(y_w, -x_w, 'k--', label='Outer Wall')
        ax.plot(y_s, -x_s, 'k--', label='Inner Wall')
        if cool_shaft:
            ax.plot(z_so, -y_so, 'r', lw=3, label='Wetted Inner Perimeter')
        ax.plot(z_wo, -y_wo, 'b', lw=3, label='Wetted Outer Perimeter')
        ax.fill(y_fill, z_fill, color='lightblue', alpha=0.3, label='Biochar Filled Region', edgecolor='none')
        ax.set_aspect('equal')
        ax.set_title(f"Wetted Perimeters (Degree of Fill = {100*f_fill:.1f}%)")
        ax.set_xlabel('z (m)')
        ax.set_ylabel('y (m)')
        ax.grid(True)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        st.pyplot(fig, use_container_width=True)
    plot_perimeters()

with st.sidebar.expander("🌡️ Cooling Water Temperatures"):
    st.markdown("### Cooling Water Inlet Temperature")
    st.markdown(f"{t_water_in} °C")
    
    st.markdown("### Cooling Water Outlet Temperatures")
    for entry in cooling_results:
        st.markdown(f"- x: {entry['x']}, Shaft Out: {entry['T_water_shaft_out (°C)']} °C, Casing Out: {entry['T_water_casing_out (°C)']} °C")
    
    st.markdown("---")  # horizontal separator
    
    st.markdown(f"**Cooling Water Velocity in Shaft and Casing:** {v_w} m/s")

# ------------------- Acknowledgement -------------------
with st.sidebar.expander("📚 Acknowledgement"):
    st.markdown("""
This application is based on the following publication:

**Paweł Regucki, Renata Krzyżyńska, Zbyszek Szeliga**  
*Mathematical model for a single screw ash cooler of a circulating fluidized bed boiler*,  
**Powder Technology**, Volume 396, Part A, 2022, Pages 50–58.  
ISSN: 0032-5910  
[DOI: 10.1016/j.powtec.2021.10.044](https://doi.org/10.1016/j.powtec.2021.10.044)  
[ScienceDirect Link](https://www.sciencedirect.com/science/article/pii/S0032591021009268)
""")



