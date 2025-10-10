import numpy as np
from math import exp, factorial

class JackCarRentalEnv:
    def __init__(self, max_cars, max_move,revenue,cost, lambda_rent1, lambda_rent2,lambda_return1, lambda_return2):
        self.max_cars = max_cars
        self.max_move = max_move
        self.grid_dim = max_cars + 1  # 1d mapping
        self.n_states = (max_cars + 1) ** 2
        self.start = (20, 20)  # initial state
        self.possible_actions = list(range(-max_move, max_move + 1))
        self.revenue = revenue
        self.cost = cost

        # poissons parameters
        self.lambda_rent1, self.lambda_rent2 = lambda_rent1, lambda_rent2
        self.lambda_return1, self.lambda_return2 = lambda_return1, lambda_return2

    def poisson_pmf(self, n, lam):
        """posson probability mass function"""
        return (lam ** n) * exp(-lam) / factorial(n)

    def step(self, state, action):
        """
        one step:
        - move cars
        - compute expected rewaard
        - return reward and state
        """
        cars1, cars2 = state

        # 1. move cars
        moved = min(action, cars1) if action > 0 else min(-action, cars2)
        moved = np.clip(moved, -self.max_move, self.max_move)
        cars1 -= moved
        cars2 += moved
        move_cost = abs(moved) * self.cost

        # 2. compute reward
        expected_rentals1 = sum([min(cars1, n) * self.poisson_pmf(n, self.lambda_rent1) for n in range(0, 11)])
        expected_rentals2 = sum([min(cars2, n) * self.poisson_pmf(n, self.lambda_rent2) for n in range(0, 11)])
        reward = self.revenue * (expected_rentals1 + expected_rentals2) - move_cost

        # 3. aggiorna macchine con ritorni attesi
        expected_returns1 = self.lambda_return1
        expected_returns2 = self.lambda_return2
        cars1 = min(cars1 - expected_rentals1 + expected_returns1, self.max_cars)
        cars2 = min(cars2 - expected_rentals2 + expected_returns2, self.max_cars)

        next_state = (int(cars1), int(cars2))
        done = False
        return next_state, reward, done
