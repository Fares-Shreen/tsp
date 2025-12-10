import numpy as np
import random


class AntColonyOptimizer:
    def __init__(self, num_cities, alpha=1.0, beta=2.0, evaporation_rate=0.5):
        """
        Args:
            num_ants (int): Number of ants in the colony.
            num_cities (int): Total number of cities.
            alpha (float): Importance of Pheromone (History).
            beta (float): Importance of Heuristic (Distance/Cost).
            evaporation_rate (float): How fast pheromones disappear (0.0 - 1.0).
        """
        self.num_cities = num_cities
        self.num_ants = num_cities
        self.alpha = alpha
        self.beta = beta
        self.rho = evaporation_rate

        # Pheromone Matrix: Starts with 1.0 everywhere
        self.pheromone = np.ones((num_cities, num_cities))

    def solve(self, dist_mat, vel_mat, traffic_mat, consum_mat, time_w, iterations=100):
        """
        Main ACO engine using proper inputs.

        Args:
            dist_mat: Distance between cities
            vel_mat: Velocity between cities
            traffic_mat: Traffic delays
            consum_mat: Fuel consumption
            time_w: Slider weight for time vs fuel
            iterations: Number of generations

        Returns:
            best_path, best_cost
        """
        if time_w < 0 or time_w > 1:
            return "time weight must be between 0 and 1"

        # Step 1: Calculate heuristic matrix
        heuristic_matrix = self._calculate_heuristic(dist_mat, vel_mat, traffic_mat, consum_mat, time_w)
        # Fuel Price
        fuel = 20

        # Calculate Formulas
        time_mat = (dist_mat / (vel_mat + 1e-9)) + traffic_mat
        cost_mat = consum_mat * dist_mat * fuel

        # Step 2: Initialize best path and cost
        best_path = None
        best_cost = float('inf')

        # Step 3: Run generations
        for gen in range(iterations):

            # ### NEW: Create lists to store info for later
            all_paths = []
            all_costs = []

            # Step 4: Each ant constructs a path
            for _ in range(self.num_ants):
                path = self._construct_path(heuristic_matrix)
                cost , time = self._evaluate_path_cost(path, time_mat, cost_mat, time_w)

                # ### CHANGE 1: Save path instead of depositing immediately
                all_paths.append(path)
                all_costs.append(cost)

                # Step 5: Update best path (This stays same)
                if cost < best_cost:
                    best_cost = cost
                    best_path = path

            # ### CHANGE 2: Evaporate BEFORE depositing new scent
            self.pheromone *= (1.0 - self.rho)

            # ### CHANGE 3: Batch Deposit
            # Now we update the map for everyone at the same time
            for path, cost in zip(all_paths, all_costs):
                self._deposit_pheromone(path, cost)

        return best_path, best_cost ,time

    def _calculate_heuristic(self, dist_mat, vel_mat, traffic_mat, consum_mat, time_weight):
        """
        Combines Time and Fuel into a single probability factor (Eta) to find the wanted path.
        High Heuristic = Very Desirable Road.
        """
        fuel_weight = 1.0 - time_weight

        # Fuel Price
        fuel = 20

        # Calculate Formulas
        time_mat = (dist_mat / vel_mat+ 1e-9) + traffic_mat
        Test = (dist_mat / vel_mat)
        cost_mat = consum_mat * dist_mat * fuel

        # Normalize matrices to 0-1 range so one doesn't dominate the other
        # Add epsilon (1e-9) to avoid division by zero
        norm_time = time_mat / (time_mat.max() + 1e-9)
        norm_fuel = cost_mat / (cost_mat.max() + 1e-9)

        # Weighted Cost
        weighted_cost = (norm_time * time_weight) + (norm_fuel * fuel_weight)

        # Heuristic is 1 / Cost (because ants like Low Cost)
        # We add a tiny number to avoid 1/0 errors
        return 1.0 / (weighted_cost + 1e-9)

    def _construct_path(self, heuristic_matrix):
        """
        A single ant walks through the graph.
        """
        # Start at a random city
        start_node = random.randint(0, self.num_cities - 1)
        path = [start_node]
        visited = set(path)

        for _ in range(self.num_cities - 1):
            current = path[-1]

            # Calculate Probabilities for all neighbors
            probs = self._calculate_probabilities(current, visited,
                                                  heuristic_matrix)  # heuristic_matrix is the matrix from the heuristic function

            # Roulette Wheel Selection (Pick next city based on probability)
            next_city = self._roulette_wheel_selection(probs)

            path.append(next_city)
            visited.add(next_city)

        # Return to start (Depot) to close the loop (byrg3 lnfs elmakan tany 3ashan myb2ash one-way trip)
        path.append(start_node)
        return path

    def _calculate_probabilities(self, current_city, visited, heuristic_matrix):
        """
        ACO Probability Formula: P = (Pheromone^alpha) * (Heuristic^beta)
                                      _____tau_____     ____eta_____
        """
        probabilities = np.zeros(self.num_cities)

        for city in range(self.num_cities):
            if city in visited:
                probabilities[city] = 0.0  # Can't visit twice
            else:
                tau = self.pheromone[current_city][city] ** self.alpha
                eta = heuristic_matrix[current_city][city] ** self.beta
                probabilities[city] = tau * eta

        # Normalize probabilities to sum to 1
        total = probabilities.sum()
        if total == 0:
            # If ant is stuck (shouldn't happen in fully connected graph), pick random unvisited
            unvisited = [c for c in range(self.num_cities) if c not in visited]
            probs = np.zeros(self.num_cities)
            for u in unvisited:
                probs[u] = 1.0 / len(unvisited)
            return probs

        return probabilities / total

    def _roulette_wheel_selection(self, probabilities):
        """
        Picks an index based on the probability array.
        """
        return np.random.choice(range(self.num_cities), p=probabilities)

    def _deposit_pheromone(self, path, cost):
        """
        Adds pheromone to the edges in the path.
        Amount = 1.0 / Cost
        """
        deposit_amount = 1.0 / (cost + 1e-9)
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            self.pheromone[u][v] += deposit_amount
            self.pheromone[v][u] += deposit_amount  # Symmetric graph

    def _evaluate_path_cost(self, path, time_mat, fuel_mat, time_w):
        total_time = 0.0
        total_fuel = 0.0

        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            total_time += time_mat[u][v]
            total_fuel += fuel_mat[u][v]
        fuel_price = 20
        cost = total_fuel*fuel_price
        return cost , total_time