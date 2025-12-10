import streamlit as st
import streamlit.components.v1 as components # <--- NEW IMPORT FOR HTML
import time
import matplotlib.pyplot as plt
import sys
import os
import random
import numpy as np


current_file_path = os.path.abspath(__file__)
gui_dir = os.path.dirname(current_file_path)
root_dir = os.path.dirname(gui_dir)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

try:
    from alg.algorithm import AntColonyOptimizer
    from env.environment import City, Map, generate_random_cities, visualize_cities, create_animation
except ModuleNotFoundError as e:
    st.error(f"Error importing modules: {e}")
    st.info(f"Python is looking in: {sys.path[0]}")
    st.stop()



if 'seed' not in st.session_state:
    st.session_state.seed = 0
def static_cities_optimization_process(myTimeWeight,seed):
    plt.clf()
    random.seed(seed)
    np.random.seed(seed)
    cities = generate_random_cities(10)
    world = Map(cities)
    dist_mat = world.distance_matrix
    vel_mat = world.velocity_matrix
    traffic_mat = world.traffic_matrix
    consum_mat = world.consumption_matrix
    
    aco = AntColonyOptimizer(
        num_cities=len(cities),
        alpha=1.0,
        beta=2.0,
        evaporation_rate=0.5
    )
    best_path, best_cost, time_val = aco.solve(dist_mat, vel_mat, traffic_mat, consum_mat, myTimeWeight, iterations=100)
    
    visualize_cities(world, best_path)
    static_graph = plt.gcf()

    anim_html = create_animation(world, best_path)
    
    return time_val, best_cost, static_graph, anim_html

st.set_page_config(page_title="Optimization Tool", layout="centered",)
st.title("🚚 Logistics Route Optimization Tool")

st.markdown("""
<style>
    .fuel-label { color: #2e5e4e; font-weight: bold; font-size: 16px; text-align: left; }
    .fuel-sub { color: #555; font-size: 14px; }
    .time-label { color: #2e3b4e; font-weight: bold; font-size: 16px; text-align: right; }
    .time-sub { color: #555; font-size: 14px; text-align: right; }
    .stSlider { padding-top: 0px; padding-bottom: 20px; }
</style>
""", unsafe_allow_html=True)

st.subheader("Optimization Preference")

col1, col2 = st.columns([1, 1])
with col1:
    st.markdown("<div class='fuel-label'>⛽ FUEL EFFICIENCY</div><div class='fuel-sub'>(Cost Savings)</div>", unsafe_allow_html=True)
with col2:
    st.markdown("<div style='text-align: right;'><span class='time-label'>TIME PRIORITY 🕒</span><br><span class='time-sub'>(Urgent Delivery)</span></div>", unsafe_allow_html=True)

optimization_value = st.slider(label="", min_value=0.0, max_value=1.0, value=0.5, step=0.1)

fuel_pct = int(optimization_value * 100)
time_pct = int((1 - optimization_value) * 100)

if optimization_value == 0.5:
    status_text = "Balanced Approach (50% Fuel / 50% Time)"
elif optimization_value < 0.5:
    status_text = f"Prioritizing Fuel ({fuel_pct}% Fuel / {time_pct}% Time)"
else:
    status_text = f"Prioritizing Time ({fuel_pct}% Fuel / {time_pct}% Time)"

st.markdown(f"<div style='text-align: center; color: #333; margin-top: -10px;'><b>{status_text}</b></div>", unsafe_allow_html=True)
st.write("---") 

if st.button("Randomize Cities", type="secondary", use_container_width=True):
    st.session_state.seed = random.randint(0, 1000)
    st.success(f"New Map Generated! (Seed: {st.session_state.seed})")

if st.button("RUN OPTIMIZATION", type="primary", use_container_width=True):
    
    estimated_time, cost, graph, anim_html = static_cities_optimization_process(optimization_value,st.session_state.seed)
    
    result_container = st.container()
    
    with result_container:
        with st.spinner('Calculating Optimal Route & Generating Simulation...'):
            time.sleep(1) 

            st.success("Analysis Complete")

            c1, c2, c3 = st.columns(3)
            c1.metric("Selected Alpha", f"{optimization_value}")
            c2.metric("Est. Cost", f"${cost:.2f}")
            c3.metric("Est. Time", f"{estimated_time:.1f} Hours")
            
            st.write("---")
            st.write("### 🗺 Optimized Route Map")
            st.pyplot(graph)
            st.write("### 🚗 Traffic Simulation")
            components.html(anim_html, height=800)
            
            st.write("### Detailed Output")
            st.json({
                "optimization_level": optimization_value,
                "fuel_priority": fuel_pct,
                "time_priority": time_pct,
                "status": "feasible"
            })            