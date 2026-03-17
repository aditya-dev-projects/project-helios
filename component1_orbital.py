"""
PROJECT HELIOS — Component 1: Orbital Mechanics & Solar Irradiance
===================================================================
What this does:
  1. Creates a real GEO (Geostationary Earth Orbit) satellite using actual aerospace math
  2. Propagates (moves) the satellite through 24 hours of orbit
  3. Detects eclipse periods (when Earth blocks the Sun)
  4. Calculates solar power output at every point in time
  5. Plots 3 graphs:
       - 24hr power output
       - Satellite ground track (position over Earth)
       - Eclipse timeline

Author: Built for Project Helios
Libraries: poliastro 0.17.0, astropy 5.3.4, numpy, matplotlib
"""

# ─────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
import warnings
warnings.filterwarnings("ignore")  # Suppress minor astropy warnings

from astropy import units as u
from astropy.time import Time
from poliastro.bodies import Earth, Sun
from poliastro.twobody import Orbit
from poliastro.util import norm


# ─────────────────────────────────────────────
# SECTION 1: CONSTANTS & SATELLITE PARAMETERS
# ─────────────────────────────────────────────

# Physics constants
SOLAR_CONSTANT    = 1361.0   # W/m² — solar irradiance at 1 AU (Earth's distance from Sun)
EARTH_RADIUS_KM   = 6371.0   # km — mean radius of Earth
GEO_ALTITUDE_KM   = 35786.0  # km — geostationary orbit altitude above equator
AU_IN_KM          = 1.496e8  # km — 1 Astronomical Unit

# Satellite solar array parameters (adjustable — try changing these)
PANEL_AREA_M2     = 10000    # m² — 100m × 100m collector area (1 hectare)
PANEL_EFFICIENCY  = 0.30     # 30% — realistic for high-grade space solar cells (GaAs triple junction)
DEGRADATION_RATE  = 0.005    # 0.5% efficiency loss per year in space radiation environment

# Simulation parameters
TIME_STEP_MIN     = 10       # minutes between each calculation step
SIMULATION_HOURS  = 24       # total simulation duration


# ─────────────────────────────────────────────
# SECTION 2: CREATE THE GEO ORBIT
# ─────────────────────────────────────────────

print("=" * 60)
print("PROJECT HELIOS — Component 1: Orbital Mechanics")
print("=" * 60)
print()
print("[1/5] Defining GEO orbit...")

# Define the epoch (starting time for simulation)
epoch = Time("2025-06-21 00:00:00", scale="utc")  # Summer solstice — max sun exposure

# Create a circular GEO orbit using Poliastro
# Keplerian elements for GEO:
#   - Semi-major axis: Earth radius + GEO altitude = 42,164 km
#   - Eccentricity: 0 (perfectly circular)
#   - Inclination: 0° (equatorial — stays above same Earth point)
geo_orbit = Orbit.circular(
    attractor=Earth,
    alt=GEO_ALTITUDE_KM * u.km,
    inc=0 * u.deg,
    epoch=epoch
)

# Print orbital parameters for verification
print(f"   Orbit type     : Geostationary (GEO)")
print(f"   Altitude       : {GEO_ALTITUDE_KM:,.0f} km")
print(f"   Semi-major axis: {geo_orbit.a.to(u.km).value:,.1f} km")
print(f"   Orbital period : {geo_orbit.period.to(u.hour).value:.2f} hours")
print(f"   Orbital speed  : 3.075 km/s")
print(f"   Inclination    : {geo_orbit.inc.to(u.deg).value:.1f}°")
print()


# ─────────────────────────────────────────────
# SECTION 3: PROPAGATE ORBIT OVER 24 HOURS
# ─────────────────────────────────────────────

print("[2/5] Propagating orbit over 24 hours...")

# Generate time steps (every 10 minutes for 24 hours = 144 steps)
time_steps_min = np.arange(0, SIMULATION_HOURS * 60, TIME_STEP_MIN)
num_steps = len(time_steps_min)

# Arrays to store results
positions_km    = []   # 3D position vectors [x, y, z] in km
longitudes_deg  = []   # satellite longitude over Earth (for ground track)
latitudes_deg   = []   # satellite latitude over Earth

for t_min in time_steps_min:
    # Propagate the orbit forward by t_min minutes
    propagated = geo_orbit.propagate(t_min * u.min)
    
    # Get position vector in km [x, y, z] in Earth-Centered Inertial (ECI) frame
    pos = propagated.r.to(u.km).value
    positions_km.append(pos)
    
    # Convert ECI position to longitude/latitude for ground track visualization
    # For GEO (0° inclination), latitude is always ~0°
    lon = np.degrees(np.arctan2(pos[1], pos[0]))  # longitude in degrees
    lat = np.degrees(np.arcsin(pos[2] / np.linalg.norm(pos)))  # latitude in degrees
    longitudes_deg.append(lon)
    latitudes_deg.append(lat)

positions_km   = np.array(positions_km)
longitudes_deg = np.array(longitudes_deg)
latitudes_deg  = np.array(latitudes_deg)

print(f"   Computed {num_steps} position samples over {SIMULATION_HOURS} hours")
print(f"   Position range X: {positions_km[:,0].min():.0f} to {positions_km[:,0].max():.0f} km")
print()


# ─────────────────────────────────────────────
# SECTION 4: ECLIPSE DETECTION
# ─────────────────────────────────────────────
# 
# Method: Cylindrical shadow model
# Concept: If the satellite is "behind" Earth relative to the Sun,
#          and within Earth's shadow cylinder, it's in eclipse.
#
# This is the standard simplified model used in preliminary mission analysis.
# (More accurate: conical penumbra/umbra model — for future improvement)
# ─────────────────────────────────────────────

print("[3/5] Computing eclipse periods...")

def get_sun_direction(time_offset_hours):
    """
    Returns a unit vector pointing from Earth toward the Sun.
    Uses Earth's orbital angle around the Sun based on day of year.
    Simplified 2D model (ignores Earth's axial tilt for eclipse detection).
    
    Parameters:
        time_offset_hours: hours since simulation start
    Returns:
        numpy array [x, y, z] — unit vector toward Sun
    """
    # Earth moves ~1° per day around the Sun (360°/365 days)
    # Start angle at summer solstice (June 21): ~172° in ecliptic longitude
    start_angle_deg = 172.0
    earth_orbital_speed = 360.0 / 365.25  # degrees per day
    
    angle_deg = start_angle_deg + (time_offset_hours / 24.0) * earth_orbital_speed
    angle_rad = np.radians(angle_deg)
    
    # Sun direction vector (Earth → Sun)
    sun_dir = np.array([np.cos(angle_rad), np.sin(angle_rad), 0.0])
    return sun_dir


def check_eclipse(satellite_pos_km, sun_direction):
    """
    Cylindrical shadow model for eclipse detection.
    
    Logic:
      1. Project satellite position onto the Earth-Sun axis
      2. If projection is negative (satellite is on night side of Earth):
         → Check if satellite is within Earth's shadow cylinder radius
      3. If within cylinder: ECLIPSE. Else: SUNLIGHT.
    
    Parameters:
        satellite_pos_km: [x, y, z] position vector in km
        sun_direction: unit vector pointing toward Sun
    Returns:
        True if in eclipse, False if in sunlight
    """
    # Dot product = projection of satellite position onto Sun direction
    # Negative value means satellite is on the anti-sun side (night side)
    projection = np.dot(satellite_pos_km, sun_direction)
    
    if projection > 0:
        # Satellite is on the Sun-facing side → definitely in sunlight
        return False
    
    # Calculate perpendicular distance from the Earth-Sun axis
    # This tells us if the satellite is within Earth's shadow cylinder
    along_component = projection * sun_direction
    perp_vector = satellite_pos_km - along_component
    perp_distance_km = np.linalg.norm(perp_vector)
    
    # If perpendicular distance < Earth's radius → in shadow
    return perp_distance_km < EARTH_RADIUS_KM


# Run eclipse detection for all time steps
eclipse_flags = []
for i, t_min in enumerate(time_steps_min):
    t_hours = t_min / 60.0
    sun_dir = get_sun_direction(t_hours)
    in_eclipse = check_eclipse(positions_km[i], sun_dir)
    eclipse_flags.append(in_eclipse)

eclipse_flags = np.array(eclipse_flags)

# Summary statistics
eclipse_minutes = np.sum(eclipse_flags) * TIME_STEP_MIN
sunlight_minutes = (num_steps - np.sum(eclipse_flags)) * TIME_STEP_MIN
uptime_percent = (sunlight_minutes / (SIMULATION_HOURS * 60)) * 100

print(f"   Total eclipse duration : {eclipse_minutes:.0f} minutes ({100-uptime_percent:.1f}%)")
print(f"   Total sunlight duration: {sunlight_minutes:.0f} minutes ({uptime_percent:.1f}%)")
print(f"   Power uptime           : {uptime_percent:.1f}% (vs ~15-20% for ground solar)")
print()


# ─────────────────────────────────────────────
# SECTION 5: CALCULATE SOLAR POWER OUTPUT
# ─────────────────────────────────────────────
#
# Physics:
#   Power = Irradiance × Area × Efficiency × cos(pointing_angle)
#
# At GEO, the satellite's solar tracking system keeps panels
# pointed directly at the Sun (pointing_angle ≈ 0° most of the time).
# Small seasonal variation of ±23.5° exists due to Earth's axial tilt.
# ─────────────────────────────────────────────

print("[4/5] Calculating power output...")

def calculate_power_watts(is_eclipsed, panel_area, efficiency,
                           pointing_offset_deg=0.0, years_in_orbit=0):
    """
    Calculate instantaneous power output of the solar array.
    
    Parameters:
        is_eclipsed         : bool — whether satellite is in Earth's shadow
        panel_area          : float — collector area in m²
        efficiency          : float — solar cell efficiency (0 to 1)
        pointing_offset_deg : float — angle between panel normal and Sun direction
        years_in_orbit      : float — mission age in years (for degradation)
    Returns:
        power in Watts
    """
    if is_eclipsed:
        return 0.0
    
    # Apply radiation degradation over mission life
    degraded_efficiency = efficiency * ((1 - DEGRADATION_RATE) ** years_in_orbit)
    
    # Cosine factor — panels produce less power if not pointing directly at Sun
    cosine_factor = np.cos(np.radians(pointing_offset_deg))
    
    # Core power equation
    power = SOLAR_CONSTANT * panel_area * degraded_efficiency * cosine_factor
    return power


# Calculate power at each time step
# Slight pointing variation: ±2° random jitter to simulate realistic tracking error
pointing_jitter = np.random.uniform(-2, 2, num_steps)

power_watts = np.array([
    calculate_power_watts(
        is_eclipsed=eclipse_flags[i],
        panel_area=PANEL_AREA_M2,
        efficiency=PANEL_EFFICIENCY,
        pointing_offset_deg=pointing_jitter[i],
        years_in_orbit=0
    )
    for i in range(num_steps)
])

power_MW = power_watts / 1e6  # Convert to Megawatts

# Summary statistics
peak_power_MW    = power_MW.max()
average_power_MW = power_MW.mean()
min_power_MW     = power_MW.min()
daily_energy_MWh = average_power_MW * SIMULATION_HOURS

print(f"   Panel area         : {PANEL_AREA_M2:,} m² ({PANEL_AREA_M2/10000:.1f} hectares)")
print(f"   Panel efficiency   : {PANEL_EFFICIENCY*100:.0f}%")
print(f"   Peak power output  : {peak_power_MW:.2f} MW")
print(f"   Average power      : {average_power_MW:.2f} MW")
print(f"   Daily energy yield : {daily_energy_MWh:.1f} MWh")
print(f"   Homes powered (est): ~{int(daily_energy_MWh / 0.03):,} homes/day")  # avg 30 kWh/day per home
print()


# ─────────────────────────────────────────────
# SECTION 6: VISUALIZATION — 3 PLOTS
# ─────────────────────────────────────────────

print("[5/5] Generating plots...")

fig = plt.figure(figsize=(16, 12))
fig.patch.set_facecolor('#0d1117')  # Dark background — space theme
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)

time_hours = time_steps_min / 60.0

# ── Plot 1: 24-Hour Power Output ──────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, :])  # Spans full top row
ax1.set_facecolor('#161b22')

# Shade eclipse periods in red
for i in range(len(eclipse_flags) - 1):
    if eclipse_flags[i]:
        ax1.axvspan(time_hours[i], time_hours[i+1],
                    alpha=0.4, color='#ff4444', label='Eclipse' if i == 0 else "")

# Power output line
ax1.plot(time_hours, power_MW,
         color='#f0c040', linewidth=2.5, label='Solar Power Output')
ax1.fill_between(time_hours, power_MW,
                 alpha=0.15, color='#f0c040')

# Reference lines
ax1.axhline(y=average_power_MW, color='#00ff88',
            linestyle='--', linewidth=1.5, alpha=0.8,
            label=f'Average: {average_power_MW:.2f} MW')
ax1.axhline(y=peak_power_MW, color='#ffffff',
            linestyle=':', linewidth=1.0, alpha=0.5,
            label=f'Peak: {peak_power_MW:.2f} MW')

ax1.set_xlabel('Time (hours)', color='#cccccc', fontsize=11)
ax1.set_ylabel('Power Output (MW)', color='#cccccc', fontsize=11)
ax1.set_title('GEO Solar Power Station — 24-Hour Power Generation Profile',
              color='#ffffff', fontsize=13, fontweight='bold', pad=12)
ax1.tick_params(colors='#cccccc')
ax1.set_xlim(0, 24)
ax1.set_ylim(0, peak_power_MW * 1.15)
ax1.grid(True, alpha=0.2, color='#ffffff')
ax1.legend(loc='lower right', facecolor='#161b22',
           labelcolor='white', fontsize=9, framealpha=0.8)

# Annotate eclipse if any occurred
if eclipse_flags.any():
    eclipse_start = time_hours[eclipse_flags][0]
    ax1.annotate('Eclipse\n(Earth shadow)',
                 xy=(eclipse_start + 0.2, peak_power_MW * 0.5),
                 color='#ff8888', fontsize=9, ha='center')

# ── Plot 2: Ground Track ───────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[1, 0])
ax2.set_facecolor('#161b22')

# Draw a simple Earth outline
theta = np.linspace(0, 2 * np.pi, 360)
ax2.plot(np.degrees(theta), np.zeros_like(theta),
         color='#444444', linewidth=0.5, alpha=0.5)

# Plot satellite track — color coded by eclipse status
for i in range(len(longitudes_deg) - 1):
    color = '#ff4444' if eclipse_flags[i] else '#f0c040'
    ax2.plot(longitudes_deg[i:i+2], latitudes_deg[i:i+2],
             color=color, linewidth=2.5, alpha=0.8)

# Mark start position
ax2.scatter(longitudes_deg[0], latitudes_deg[0],
            color='#00ff88', s=100, zorder=5, label='Start', marker='*')

ax2.set_xlabel('Longitude (°)', color='#cccccc', fontsize=10)
ax2.set_ylabel('Latitude (°)', color='#cccccc', fontsize=10)
ax2.set_title('Satellite Ground Track\n(GEO stays above equator)',
              color='#ffffff', fontsize=11, fontweight='bold')
ax2.tick_params(colors='#cccccc')
ax2.set_xlim(-180, 180)
ax2.set_ylim(-30, 30)
ax2.grid(True, alpha=0.2, color='#ffffff')
ax2.legend(facecolor='#161b22', labelcolor='white', fontsize=9)
ax2.axhline(y=0, color='#4444ff', linewidth=1, linestyle='--', alpha=0.5)

# ── Plot 3: Eclipse Timeline ───────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 1])
ax3.set_facecolor('#161b22')

# Power as bar chart to clearly show eclipse gaps
bar_colors = ['#ff4444' if e else '#f0c040' for e in eclipse_flags]
ax3.bar(time_hours, power_MW, width=TIME_STEP_MIN/60,
        color=bar_colors, alpha=0.85, align='edge')

ax3.set_xlabel('Time (hours)', color='#cccccc', fontsize=10)
ax3.set_ylabel('Power (MW)', color='#cccccc', fontsize=10)
ax3.set_title('Eclipse vs Sunlight Timeline\n(Red = Eclipse, Yellow = Sunlight)',
              color='#ffffff', fontsize=11, fontweight='bold')
ax3.tick_params(colors='#cccccc')
ax3.set_xlim(0, 24)
ax3.grid(True, alpha=0.2, color='#ffffff', axis='y')

# ── Overall title ──────────────────────────────────────────────────────────────
fig.suptitle('PROJECT HELIOS — Component 1: Orbital Mechanics & Solar Irradiance\n'
             f'GEO Altitude: {GEO_ALTITUDE_KM:,} km | Array: {PANEL_AREA_M2:,} m² | '
             f'Efficiency: {PANEL_EFFICIENCY*100:.0f}% | Peak: {peak_power_MW:.2f} MW',
             color='#ffffff', fontsize=12, fontweight='bold', y=0.98)

plt.savefig('helios_component1_output.png', dpi=150,
            bbox_inches='tight', facecolor='#0d1117')
plt.show()

print()
print("=" * 60)
print("COMPONENT 1 COMPLETE")
print("=" * 60)
print(f"Output image saved: helios_component1_output.png")
print()
print("SUMMARY:")
print(f"  Satellite altitude    : {GEO_ALTITUDE_KM:,} km (GEO)")
print(f"  Orbital period        : {geo_orbit.period.to(u.hour).value:.2f} hours (= 1 sidereal day)")
print(f"  Power uptime          : {uptime_percent:.1f}%")
print(f"  Peak power            : {peak_power_MW:.2f} MW")
print(f"  Daily energy          : {daily_energy_MWh:.1f} MWh")
print(f"  Estimated homes/day   : {int(daily_energy_MWh / 0.03):,}")
print()
print("NEXT STEP → Component 2: Microwave Transmission Loss Model (ITU-R P.676)")
print("=" * 60)