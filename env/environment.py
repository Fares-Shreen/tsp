import math
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation





# ======================================================
# ---------------------- CITY --------------------------
# ======================================================

class City:
    def __init__(self, id, x, y):
        self.id = id
        self.x = x
        self.y = y

    def distance_to(self, other_city):
        # Correct Euclidean distance: (x2 - x1)^2 + (y2 - y1)^2
        return math.sqrt((self.x - other_city.x) ** 2 + (self.y - other_city.y) ** 2)

    def __repr__(self):
        return f"City({self.id}, x={self.x:.2f}, y={self.y:.2f})"


# ======================================================
# ----------------------- MAP --------------------------
# ======================================================

class Map:
    def __init__(self, cities):
        self.cities = cities
        self.num_cities = len(cities)

        # Mandatory matrices
        self.distance_matrix = self.build_distance_matrix()
        self.velocity_matrix = self.build_velocity_matrix()
        self.traffic_matrix = self.build_traffic_matrix()
        self.consumption_matrix = self.build_consumption_matrix()

    # ---------------- Distance Matrix ------------------
    def build_distance_matrix(self):
        n = self.num_cities
        matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                matrix[i][j] = self.cities[i].distance_to(self.cities[j])

        return matrix

    # ---------------- Velocity Matrix ------------------
    def build_velocity_matrix(self):
        """
        Generate speed limits between nodes.
        Example realistic speed values (km/h):
        - City roads: 30–60
        - Highways: 80–120
        """
        n = self.num_cities
        matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i == j:
                    matrix[i][j] = -1
                else:
                    matrix[i][j] = random.choice([40, 60, 80, 100, 120])

        return matrix

    # ---------------- Traffic Matrix -------------------
    def build_traffic_matrix(self):
        """
        Random traffic delays in hours.
        Typical delays:
        - Light traffic: 0–0.3 h
        - Moderate: 0.3–0.7 h
        - Heavy: 1.0–2.0 h
        """
        n = self.num_cities
        matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i == j:
                    matrix[i][j] = 0
                else:
                    matrix[i][j] = round(random.uniform(0, 1.5), 2)

        return matrix

    # ---------------- Consumption Matrix ---------------
    def build_consumption_matrix(self):
        """
        Fuel consumption (liters/km) depends mainly on SPEED.
        """
        speed_to_consumption = {
            40: (0.06, 0.08),
            60: (0.07, 0.09),
            80: (0.08, 0.11),
            100: (0.10, 0.14),
            120: (0.12, 0.18),
        }

        n = self.num_cities
        matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                speed = int(self.velocity_matrix[i][j])
                if speed == -1:
                    matrix[i][j] = 0
                    continue

                low, high = speed_to_consumption[speed]
                matrix[i][j] = round(random.uniform(low, high), 3)

        return matrix

    def __repr__(self):
        return f"Map with {self.num_cities} cities"


# ======================================================
# ------------- DATA GENERATION FUNCTIONS --------------
# ======================================================

def generate_random_cities(n=20, x_range=(0, 100), y_range=(0, 100)):
    cities = []
    for i in range(n):
        x = random.uniform(*x_range)
        y = random.uniform(*y_range)
        cities.append(City(i, x, y))
    return cities


def load_cities_from_csv(path):
    df = pd.read_csv(path)
    cities = []

    for _, row in df.iterrows():
        cities.append(City(int(row["id"]), float(row["x"]), float(row["y"])))

    return cities


# ======================================================
# ------------------- VISUALIZATION --------------------
# ======================================================




def visualize_cities(world, path=None):
    """
    Visualize the cities stored in the Map object.
    Each city will be plotted with its index.
    All cities are connected with lines (complete graph).
    """

    xs = [city.x for city in world.cities]
    ys = [city.y for city in world.cities]

    plt.figure(figsize=(7, 7))

    # Draw lines connecting all cities to each other
    for i in range(len(world.cities)):
        for j in range(i + 1, len(world.cities)):
            x_coords = [world.cities[i].x, world.cities[j].x]
            y_coords = [world.cities[i].y, world.cities[j].y]
            plt.plot(x_coords, y_coords, 'gray', alpha=0.3, linewidth=0.5)

    # Plot cities as points
    plt.scatter(xs, ys, color='red', s=150, zorder=5)

    # Label each city
    for i, city in enumerate(world.cities):
        plt.text(city.x, city.y, str(city.id), fontsize=10, ha='center', va='center', color='white', weight='bold')

    if path is not None and len(path) > 1:
        for i in range(len(path) - 1):
            c1 = world.cities[path[i]]
            c2 = world.cities[path[i + 1]]

            print(c1,c2)
            plt.plot(
                [c1.x, c2.x],
                [c1.y, c2.y],
                color='blue', linewidth=3.0, zorder=10
            )

    print(path)
    start_city = world.cities[path[0]]
    plt.scatter(start_city.x, start_city.y, s=180, color='green', zorder=15)
    plt.text(start_city.x, start_city.y - 3, "START", color='green', fontsize=10, ha='center', weight='bold')

    for order, city_index in enumerate(path):
        # print(order, city_index)
        # print(world.cities[city_index])
        city_obj = world.cities[city_index]
        # print(city_obj.x)

        plt.text(
            city_obj.x, city_obj.y -.5,
            str(city_index), 
            color='white',
            fontsize=10,
            ha='center',
            weight='bold',
            zorder=20
        )

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("City Map Visualization (All Cities Connected)")
    path_str = " → ".join(map(str, path))
    plt.text(0.5, 1.08, f"Path: {path_str}", transform=plt.gca().transAxes,
             fontsize=12, ha='center', va='top', weight='bold', color='blue')
    plt.show()



def create_animation(world, path):
    """ Animation Visualization """
    # Setup Figure (Standard 8x6)
    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)

    # Background
    x_coords = [city.x for city in world.cities]
    y_coords = [city.y for city in world.cities]
    
    for i, city1 in enumerate(world.cities):
        for j, city2 in enumerate(world.cities):
            if i != j:
                ax.plot([city1.x, city2.x], [city1.y, city2.y], color='lightgray', linewidth=0.5, alpha=0.5)

    ax.scatter(x_coords, y_coords, c='red', s=100, zorder=5)
    for i, city in enumerate(world.cities):
        ax.text(city.x, city.y + 2, str(i), fontsize=12, ha='center')

    # Car
    start_x = world.cities[path[0]].x
    start_y = world.cities[path[0]].y
    car, = ax.plot([start_x], [start_y], 'go', markersize=15, markeredgecolor='black', zorder=20, label='Delivery Car')
    
    # Path Interpolation
    route_x = []
    route_y = []
    steps_per_segment = 20 

    for i in range(len(path) - 1):
        c1 = world.cities[path[i]]
        c2 = world.cities[path[i+1]]
        x_pts = np.linspace(c1.x, c2.x, steps_per_segment)
        y_pts = np.linspace(c1.y, c2.y, steps_per_segment)
        route_x.extend(x_pts)
        route_y.extend(y_pts)

    def update(frame):
        idx = frame % len(route_x)
        car.set_data([route_x[idx]], [route_y[idx]])
        return car,

    ax.set_title(f"Simulation")
    ax.legend(loc='upper left')
    
    # Ensure layout is tight but respects our manual limits
    plt.tight_layout()

    ani = animation.FuncAnimation(fig, update, frames=len(route_x), interval=50, blit=False, repeat=True)
    return ani.to_jshtml(default_mode='loop')