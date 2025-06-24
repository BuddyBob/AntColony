"""A civilization game where ants spawn in a civilizatiuon where they can choose 
there speed but that will affect there hunger. 
So the faster they move the faster there health depletes. 
Food will spawn randomly and ALl the ants will spawn in one place there colony and spread out 
"""

import pygame
import neat
import os
import random
import math
import pickle
import time

pygame.init()
WIDTH, HEIGHT = 1200, 1200
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Ant Colony")
clock = pygame.time.Clock()

font = pygame.font.SysFont(None, 28)

#CONSTANTS
FOOD_SPAWN_TIME = 30
MAX_FOOD = 20
FOOD_WORTH = 600
MIN_FOOD_REPLUSION = 1
HUNGER_DECAY = .1
HEALTH_DECAY_EMPTY = 2 # when hunger is 0
MAX_HUNGER, MAX_HEALTH =  300, 200
PICKUP_RADIUS = 30
SPEED_LEVELS = [1,1.2,1.4]
MAX_FRAMES = 1000 #1 minute of simulation

class Ant:
    def __init__(self, x_pos, y_pos, size=7):
        self.X_POS = x_pos
        self.Y_POS = y_pos
        self.SIZE = size
        self.color = (201, 179, 50)
        self.ant_rect = pygame.Rect(self.X_POS, self.Y_POS, self.SIZE, self.SIZE)
        self.hunger = MAX_HUNGER
        self.health = MAX_HEALTH
        self.speed_level = 1
        self.carrying_food = False

    def draw(self, screen):
        pygame.draw.rect(screen, self.color, self.ant_rect)

    def hunger_health(self):
        self.hunger -= HUNGER_DECAY + self.speed_level
        if self.hunger <= 0:
            self.hunger = 0
            self.carrying_food = False
            self.health -= HEALTH_DECAY_EMPTY
            if self.health <= 0:
                self.health = 0


    def move(self, dx, dy):
        self.ant_rect.x += dx*self.speed_level
        self.ant_rect.y += dy*self.speed_level
        # Keep ant within bounds
        if self.ant_rect.x < 0:
            self.ant_rect.x = 0
        elif self.ant_rect.x > WIDTH - self.SIZE:
            self.ant_rect.x = WIDTH - self.SIZE
        if self.ant_rect.y < 0:
            self.ant_rect.y = 0
        elif self.ant_rect.y > HEIGHT - self.SIZE:
            self.ant_rect.y = HEIGHT - self.SIZE

    def eat(self):
        if self.carrying_food:
            self.hunger += FOOD_WORTH
            self.health += FOOD_WORTH // 2
            if self.hunger > MAX_HUNGER:
                self.hunger = MAX_HUNGER
                self.health = MAX_HEALTH 
            self.carrying_food = False

        self.hunger = min(self.hunger, MAX_HUNGER)
        self.health = min(self.health, MAX_HEALTH)

class World:
    def __init__(self):
        self.food_positions = []
        self.last_food_spawn_time = 0

    def spawn_food(self):
        if len(self.food_positions) < MAX_FOOD:
            x = random.randint(200, WIDTH-200)
            y = random.randint(200, HEIGHT-200)
            self.food_positions.append((x, y))

    def draw_food(self, screen):
        for pos in self.food_positions:
            pygame.draw.circle(screen, (171, 60, 55), pos, 4)

def eval_genomes(genomes, config):
    ants, nets, ge = [], [], []

    #create multiple spawn points
    BORDER, rand_edge = 80, lambda: [(random.randint(BORDER, WIDTH-BORDER), 0), (random.randint(BORDER, WIDTH-BORDER), HEIGHT), (0, random.randint(BORDER, HEIGHT-BORDER)), (WIDTH, random.randint(BORDER, HEIGHT-BORDER))][random.randrange(4)]
    NESTS_POS = [rand_edge() for _ in range(10)]
    
    # Create ants at nest positions
    for idx, (gid, genome) in enumerate(genomes):
        genome.fitness = 0
        nets.append(neat.nn.FeedForwardNetwork.create(genome, config))
        nest_x, nest_y = NESTS_POS[idx % len(NESTS_POS)]   
        ants.append(Ant(nest_x, nest_y))
        ge.append(genome)


    run = True
    food_spawn_timer = 0

    #Initialize the world with some food
    world = World()
    for i in range(MAX_FOOD):  # seed map with 10 pellets
        world.spawn_food()

    frame = 0
    while run and ants and frame < MAX_FRAMES:
        frame += 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()
        screen.fill((41, 40, 38))


        #Spawn the food
        food_spawn_timer += 1
        if food_spawn_timer >= FOOD_SPAWN_TIME:
            world.spawn_food()
            food_spawn_timer = 0
        world.draw_food(screen)



        for i in reversed(range(len(ants))): 
            ant = ants[i]
            nearest_food = None
            min_dist = float('inf')

            # Find nearest food
            for fx, fy in world.food_positions:
                dist = math.hypot(fx - ant.ant_rect.x, fy - ant.ant_rect.y)
                if dist < min_dist:
                    min_dist = dist
                    nearest_food = (fx, fy)
            
            if nearest_food is None:
                continue  

            # Inputs: Ant position, nearest food position, hunger, health, carrying food
            # Normalize positions and distances
            dx = (nearest_food[0] - ant.ant_rect.x)/ WIDTH       
            dy = (nearest_food[1] - ant.ant_rect.y)/ HEIGHT     
            dist = min_dist/math.hypot(WIDTH, HEIGHT)      

            inputs = [
                dx, dy, dist,
                ant.hunger/MAX_HUNGER,
                ant.health/MAX_HEALTH,
                int(ant.carrying_food)
            ]

            outputs = nets[i].activate(inputs)
            move_x = (outputs[0] - 0.5) * 2 
            move_y = (outputs[1] - 0.5) * 2
            speed_choice = outputs[2]

            # Update speed level based on output
            if speed_choice < 0.33:
                ant.speed_level = 1
            elif speed_choice < 0.66:
                ant.speed_level = 2
            else:
                ant.speed_level = 3




            # Check if we should pick up food
            prev_dist = min_dist
            if not ant.carrying_food and nearest_food:
                if math.hypot(nearest_food[0] - ant.ant_rect.x, nearest_food[1] - ant.ant_rect.y) < PICKUP_RADIUS:
                    ant.carrying_food = True
                    ant.eat()
                    world.food_positions.remove(nearest_food) #remove food from world
                    world.spawn_food() #spawn new food
                    ge[i].fitness += 20
                    ant.carrying_food = False

            ant.move(move_x, move_y)
            ant.hunger_health()
            ant.draw(screen)

            # Check if ant is dead
            if ant.health <= 0 or ge[i].fitness < 0:
                ge[i].fitness -= 1
                ants.pop(i)
                nets.pop(i)
                ge.pop(i)
                continue



            # - -- -  - -- Fitness Logic - - - - - - #
            def clamp(val, lo, hi):          
                return max(lo, min(val, hi))

            h_ratio = ant.hunger / MAX_HUNGER     
            b_ratio = ant.health / MAX_HEALTH              
            ge[i].fitness += 0.04 * (0.6*h_ratio + 0.4*b_ratio)

            new_dist = math.hypot(nearest_food[0] - ant.ant_rect.x,nearest_food[1] - ant.ant_rect.y) if nearest_food else 0
            delta = clamp(prev_dist - new_dist, -10, 10) 

            if frame % 15 == 0:
                ge[i].fitness += 0.1 * delta              




            # starvation penalty every 30 ticks
            if frame % 30 == 0:
                if ant.hunger < MAX_HUNGER * 0.2:
                    ge[i].fitness -= 0.3

                # idle penalty
                if abs(move_x) + abs(move_y) < 0.1:
                    ge[i].fitness -= 0.01



            ge[i].fitness = max(0, ge[i].fitness)

        # HUD
        best_now = max(g.fitness for g in ge) if ge else 0
        gen_lbl   = font.render(f"Gen: {p.generation}", True, (0, 0, 0))
        fit_lbl   = font.render(f"Best fitness: {best_now:.1f}", True, (0, 0, 0))
        ants_lbl  = font.render(f"Ants alive: {len(ants)}", True, (0, 0, 0))
        best_hunger = max(ant.hunger for ant in ants) if ants else 0
        hunger_lbl = font.render(f"Best Hunger: {best_hunger}", True, (0, 0, 0))
        best_health = max(ant.health for ant in ants) if ants else 0
        health_lbl = font.render(f"Best Health: {best_health}", True, (0, 0, 0))


        screen.blit(gen_lbl, (10, 10))
        screen.blit(fit_lbl, (10, 40))
        screen.blit(ants_lbl, (10, 70)) 
        screen.blit(hunger_lbl, (10, 100))
        screen.blit(health_lbl, (10, 130))

        # Draw nests
        for nest_pos in NESTS_POS:
            pygame.draw.circle(screen, (28, 27, 27), nest_pos, 15)

        clock.tick(30)
        pygame.display.update()


def run_neat(config):
    global p
    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config
    )
    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    #p.add_reporter(neat.Checkpointer(5))

    winner = p.run(eval_genomes, 50)

    with open("./best_ant_genome.pkl", "wb") as f:
        pickle.dump(winner, f)
    print("\nBest genome saved to best_ant_genome.pkl")


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config.txt")
    run_neat(config_path)