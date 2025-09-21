# Ant Colony – NEAT-Powered Simulation

This project implements an **ant colony simulation** driven by the **NEAT (NeuroEvolution of Augmenting Topologies)** algorithm.  
Agents (ants) evolve control policies to locate food, return it to their nests, and balance the trade-off between **movement speed and energy expenditure**.  
Faster ants can reach food more quickly but deplete hunger and health at a higher rate, creating evolutionary pressure toward efficient strategies.

---

## Simulation Overview
- **Ant coloring:** Ant color corresponds to speed (green = slower, red = faster).  
- **Food collection:** Ants carrying food are marked in white.  
- **Environment markers:** Red dots represent food items; black dots represent nest locations.  
- **Behavioral dynamics:** Agents must manage hunger decay, health loss, and survival while optimizing food retrieval.

---

## Technical Features
- **Neuroevolutionary learning:** Control policies are evolved using NEAT feed-forward networks with continuous movement outputs.  
- **Resource and survival system:** Hunger and health values decay over time, with decay rates scaled by movement speed; starvation leads to agent death.  
- **Fitness evaluation:** Rewards for collecting food, returning to nests, maintaining higher hunger/health, and efficient movement; penalties for starvation and inactivity.  
- **Dynamic environment:** Multiple nest locations, limited food capacity, and periodic food spawning introduce resource scarcity and competition.  
- **Visualization and metrics:** Pygame-based interface with live statistics including generation, best fitness, average speed, hunger, and health.

---

## Key Configuration Parameters
```python
MAX_FOOD = 10        # maximum number of food items on screen
FOOD_SPAWN_TIME = 30 # frames between new food spawns
MAX_FRAMES = 1000    # maximum number of frames per simulation (~1 minute)
ANT_COUNT = 100      # number of ants evaluated in each generation
