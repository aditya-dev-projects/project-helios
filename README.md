# Project Helios — Space Based Solar Power Simulation

A physics-based simulation of a Space-Based Solar Power (SBSP) system
built in Python using real aerospace engineering standards.

## Components Built

### Component 1 — Orbital Mechanics & Solar Irradiance
- GEO satellite orbit using Poliastro (real aerospace math)
- Eclipse detection via cylindrical shadow model
- Solar power output: **4.08 MW peak** from 10,000 m² array
- Power uptime: **95.1%** vs 15-20% for ground solar

### Component 2 — Microwave Transmission Loss Model
- ITU-R P.676 atmospheric absorption standard
- ITU-R P.838 rain attenuation standard  
- End-to-end efficiency: **70.5%**
- Delivered to ground: **2.88 MW**

## Tech Stack
- Python 3.10
- Poliastro 0.17.0 — orbital mechanics
- Astropy 5.3.4 — astronomy calculations
- NumPy, Matplotlib, Scipy

## Run It
```bash
pip install poliastro==0.17.0 astropy==5.3.4 numpy matplotlib scipy
python component1_orbital.py
python component2_transmission.py
```

## Project Vision
Full simulation will include Component 3 (rectenna model),
Component 4 (AI optimization), and Component 5 (Streamlit dashboard).

Built by Aditya Patil — 6th Sem AIML Engineering