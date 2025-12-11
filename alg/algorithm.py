import numpy as np
import random

class AntColonyOptimizer:
    def __init__(self, num_cities, alpha=1.0, beta=2.0, evaporation_rate=0.5, num_ants=None):
        """
        Args:
            num_cities (int): Total number of cities.
            alpha (float): Importance of Pheromone (history).
            beta (float): Importance of Heuristic (distance/cost).
            evaporation_rate (float): How fast pheromones disappear (0.0 - 1.0).
        """
        self.num_cities = num_cities
        self.num_ants = num_cities + 40
        self.alpha = alpha
        self.beta = beta
        self.rho = evaporation_rate

        # Pheromone Matrix: Starts with 1.0 everywhere
        self.pheromone = np.ones((num_cities, num_cities))

    def solve(self, dist_mat, vel_mat, traffic_mat, consum_mat, time_w, iterations=100, fuel_price=20):
        """
        Main ACO engine.

        Args:
            dist_mat: matrix of distances.
            vel_mat: matrix of expected velocities.
            traffic_mat: matrix of traffic delays (time).
            consum_mat: matrix of fuel consumption per distance unit (e.g. liters/km).
            time_w: slider weight between 0 and 1 (1 = prioritize time, 0 = prioritize fuel cost).
            iterations: number of generations.
            fuel_price: price per fuel unit (e.g. $/liter)

        Returns:
            best_path (list), best_cost (float), best_time (float)
        """
        if time_w < 0 or time_w > 1:
            raise ValueError("time weight must be between 0 and 1")

        eps = 1e-9

        # Compute time matrix and fuel consumption/cost matrices consistently here:
        time_mat = (dist_mat / (vel_mat + eps)) + traffic_mat
        fuel_consumption_mat = consum_mat * dist_mat
        cost_mat = fuel_consumption_mat * fuel_price

        # Store max values for consistent normalization
        self.max_time = time_mat.max() + eps
        self.max_cost = cost_mat.max() + eps
        self.time_weight = time_w
        self.fuel_weight = 1.0 - time_w

        # Build heuristic from edge-level matrices
        heuristic_matrix = self._calculate_heuristic_from_time_and_cost(time_mat, cost_mat, time_w)

        # Initialize best path and cost/time
        best_path = None
        best_weighted_cost = float('inf')
        best_time = 0
        best_fuel_cost = 0

        for gen in range(iterations):
            all_paths = []
            all_weighted_costs = []

            # Each ant constructs a path
            for _ in range(self.num_ants):
                path = self._construct_path(heuristic_matrix)
                weighted_cost, total_time, total_fuel_cost = self._evaluate_path_cost(
                    path, time_mat, cost_mat, time_w
                )

                all_paths.append(path)
                all_weighted_costs.append(weighted_cost)

                # Update best based on weighted cost
                if weighted_cost < best_weighted_cost:
                    best_weighted_cost = weighted_cost
                    best_path = path
                    best_time = total_time
                    best_fuel_cost = total_fuel_cost

            # Evaporate pheromones
            self.pheromone *= (1.0 - self.rho)

            # Deposit pheromone for each ant
            for path, weighted_cost in zip(all_paths, all_weighted_costs):
                self._deposit_pheromone(path, weighted_cost + eps)

        return best_path, best_fuel_cost, best_time

    def _calculate_heuristic_from_time_and_cost(self, time_mat, cost_mat, time_weight):
        """
        Heuristic from normalized time and cost matrices (EDGE-LEVEL).
        time_weight: [0,1] how important time is (1 => fully time, 0 => fully cost)
        """
        fuel_weight = 1.0 - time_weight
        eps = 1e-9

        # Normalize edges to [0,1]
        norm_time = time_mat / (time_mat.max() + eps)
        norm_cost = cost_mat / (cost_mat.max() + eps)

        # Weighted cost per edge
        weighted_cost = (norm_time * time_weight) + (norm_cost * fuel_weight)

        # Heuristic: ants prefer lower weighted cost => eta = 1 / weighted_cost
        return 1.0 / (weighted_cost + eps)

    def _construct_path(self, heuristic_matrix):
        """
        A single ant builds a Hamiltonian cycle starting at city 0.
        """
        start_node = 0
        path = [start_node]
        visited = set(path)

        for _ in range(self.num_cities - 1):
            current = path[-1]
            probs = self._calculate_probabilities(current, visited, heuristic_matrix)
            next_city = self._roulette_wheel_selection(probs)
            path.append(next_city)
            visited.add(next_city)

        # Return to start to close loop
        path.append(start_node)
        return path

    def _calculate_probabilities(self, current_city, visited, heuristic_matrix):
        """
        P = (tau^alpha) * (eta^beta)
        """
        probabilities = np.zeros(self.num_cities)

        for city in range(self.num_cities):
            if city in visited:
                probabilities[city] = 0.0
            else:
                tau = self.pheromone[current_city, city] ** self.alpha
                eta = heuristic_matrix[current_city, city] ** self.beta
                probabilities[city] = tau * eta

        total = probabilities.sum()
        if total <= 0:
            # uniform among unvisited
            unvisited = [c for c in range(self.num_cities) if c not in visited]
            probs = np.zeros(self.num_cities)
            if len(unvisited) == 0:
                return probs
            for u in unvisited:
                probs[u] = 1.0 / len(unvisited)
            return probs

        return probabilities / total

    def _roulette_wheel_selection(self, probabilities):
        """
        Pick an index with probability p.
        """
        return np.random.choice(range(self.num_cities), p=probabilities)

    def _deposit_pheromone(self, path, weighted_cost):
        """
        Adds pheromone to the edges in the path.
        deposit_amount = 1 / weighted_cost (lower cost -> more pheromone)
        """
        deposit_amount = 1.0 / weighted_cost
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            self.pheromone[u, v] += deposit_amount
            self.pheromone[v, u] += deposit_amount

    def _evaluate_path_cost(self, path, time_mat, cost_mat, time_w):
        """
        Evaluates a complete path and returns:
        - weighted_cost: Combined metric used for optimization
        - total_time: Raw time in hours
        - total_fuel_cost: Raw fuel cost in dollars
        
        CRITICAL: This must match the heuristic calculation logic!
        """
        eps = 1e-9
        fuel_w = 1.0 - time_w
        
        # Sum up raw values along the path
        total_time = 0.0
        total_fuel_cost = 0.0

        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            total_time += time_mat[u, v]
            total_fuel_cost += cost_mat[u, v]

        # Normalize path totals using the SAME max values from the matrices
        # This ensures consistency with the heuristic
        norm_time = total_time / (time_mat.max() * self.num_cities + eps)
        norm_cost = total_fuel_cost / (cost_mat.max() * self.num_cities + eps)

        # Calculate weighted cost (same formula as heuristic)
        weighted_cost = (norm_time * time_w) + (norm_cost * fuel_w)

        # Return weighted cost for optimization, plus raw values for display
        return weighted_cost, total_time, total_fuel_cost