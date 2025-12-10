from alg.algorithm import AntColonyOptimizer
from env.environment import City, Map, generate_random_cities,visualize_cities


cities = generate_random_cities(10)
world = Map(cities)
dist_mat = world.distance_matrix
vel_mat = world.velocity_matrix
traffic_mat = world.traffic_matrix
consum_mat = world.consumption_matrix
print("Distance Matrix:\n", dist_mat)
print("Velocity Matrix:\n", vel_mat)
print("Traffic Matrix:\n", traffic_mat)
print("Consumption Matrix:\n", consum_mat)

aco = AntColonyOptimizer(
    num_cities=len(cities),
    alpha=1.0,
    beta=2.0,
    evaporation_rate=0.5
)
time_weight = 0   
best_path, best_cost,time = aco.solve(dist_mat,vel_mat,traffic_mat,consum_mat,time_weight,iterations=100)
print("\nBest Path:", best_path)
print("Cost:", best_cost)
print("Time:", time)
visualize_cities(world,best_path)