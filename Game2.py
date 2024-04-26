import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np


pygame.init()
font = pygame.font.Font('arial.ttf', 25)

class Direction(Enum):
    Right = 1
    Left = 2
    Up = 3
    Down = 4

WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)
SPEED = 5000

class PuzzleroomAI:
    
    def __init__(self, w=1000, h=500):
        self.w = w
        self.h = h
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Puzzle')
        self.clock = pygame.time.Clock()
        #variables init
        self.reset()
        

    def reset(self):

        self.angle = 270
        self.circlex =self.w/3
        self.circley =self.h/2
        self.r =30
        self.reward = 0
        self.left = False
        self.right = False
        self.forward = True
        self.puzzlecx = 600
        self. puzzlecy = 70
        self.boxx = self.w*0.9
        self.boxy = self.h /2
        self.boxh = 50
        self.frame_iteration=0
        self.ballcol = False
        self.ballcolonce = False
        self.lowervec = [0,0]
        self.dist =0
        self.distreward=0
        self.initialdistbox = pygame.math.Vector2(self.puzzlecx,self.puzzlecy).distance_to((self.boxx+25,self.boxy))
        self.initialdistcircle= pygame.math.Vector2(self.puzzlecx,self.puzzlecy).distance_to((self.boxx+25,self.boxy))

        
        # init game state
        
    def play_step(self,action):
        self.frame_iteration +=1
        # collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        # 2. move
        self._move(action)
        #left,right
        # 3. check if game over
        game_over = False

        if self._is_collision() or self.frame_iteration > 1000:
            self.reward -=10
            if self.ballcolonce == True:
                self.reward +=10
                
                
            game_over = True
            return self.reward ,game_over,self.frame_iteration
            
        self._update_ui()
        self.clock.tick(SPEED)
        return self.reward, game_over, self.frame_iteration
    
    def _is_collision(self):
        #wall collisions player:
        collision = False
        if (self.circlex - self.r <= 0):
            self.circlex = 0+self.r

        if (self.circley - self.r <= 0):
            self.circley = 0+self.r

        if (self.circlex + self.r >= self.w):
                self.circlex = self.w -self.r

        if (self.circley + self.r >= self.h):
            self.circley = self.h -self.r

        #wall collisions puzzleball:
        if (self.puzzlecx - self.r <= 0):
            self.puzzlecx = 0+self.r
            collision = True
        if (self.puzzlecy - self.r <= 0):
            self.puzzlecy = 0+self.r
        if (self.puzzlecx + self.r >= self.w):
            self.puzzlecx = self.w -self.r
        if (self.puzzlecy + self.r >= self.h):
            self.puzzlecy = self.h -self.r


        #ball collide ball
        self.dist = pygame.math.Vector2(self.circlex,self.circley).distance_to((self.puzzlecx,self.puzzlecy))
        dx =  self.circlex- self.puzzlecx
        dy =   self.circley -self.puzzlecy

        distance = pygame.math.Vector2(dx,dy)
        distance = pygame.math.Vector2(distance).rotate(180)

        if self.dist < self.r * 2:
            self.puzzlecx = self.puzzlecx + distance[0]/9
            self.puzzlecy = self.puzzlecy + distance[1]/9
            self.ballcol = True
            self.ballcolonce = True
            self.reward  += 0.1
            pass
        else:
            self.ballcol = False

        #puzzleball collide with goal
        distbox = pygame.math.Vector2(self.puzzlecx,self.puzzlecy).distance_to((self.boxx+25,self.boxy))
        if self.distreward == self. initialdistbox - distbox:
            pass
        else:
            self.distreward = self. initialdistbox - distbox
            self.reward +=self. distreward/10
        if distbox < 25 + self.r:
            self.reward *2
            collision = True
            return collision
        
    def _update_ui(self):
        
        self.display.fill(BLACK)
        #Player circle
        pygame.draw.circle(self.display,RED,(self.circlex,self.circley),self.r)

        pygame.draw.line(self.display,BLACK,(self.circlex,self.circley),(self.nextpointx,self.nextpointy), 5)
        #goal
        pygame.draw.rect(self.display,BLUE2,(self.boxx,self.boxy-25, 50,self.boxh))

        #Puzzle piece circle
        pygame.draw.circle(self.display,BLUE1,(self.puzzlecx,self.puzzlecy),self.r)

        #ending
        pygame.display.flip()
        
    def _move(self,action):
        
        if np.array_equal(action,[1,0,0]):
            self.left = True
            self.right = False
            self.forward = False

        if np.array_equal(action,[0,1,0]):
            self.right = True
            self.left = False
            self.forward=False

        #calculate movement here
        if np.array_equal(action,[0,0,1]):
            self.forward = True
            self.right = False
            self.left = False
        
        vec = pygame.math.Vector2(0,self.r).rotate(self.angle)
        self.lowervec = pygame.math.Vector2(0,3).rotate(self.angle)
        self.nextpointx , self.nextpointy = self.circlex + vec.x, self.circley + vec.y
        if self.left ==True:
            self.angle -=10
        if self.right == True:
            self.angle +=10
        if self.forward == True:
            self.circlex = (self.circlex +self.lowervec.x)
            self.circley = (self.circley+self.lowervec.y)
    import numpy as np

    def discretize_coordinate(self,coord, min_val, max_val, num_bins):
        # Calculate bin size
        bin_size = (max_val - min_val) / num_bins
        
        # Determine bin index
        bin_index = int((coord - min_val) // bin_size)
        
        # Ensure bin index is within range [0, num_bins - 1]
        bin_index = np.clip(bin_index, 0, num_bins - 1)
        
        return bin_index
    
    def discretize_direction(self, vector, num_bins):
        # Calculate direction angle in degrees (-180 to 180)
        direction_angle = np.degrees(np.arctan2(vector[0], vector[1])) + 180
        
        # Convert negative angles to positive equivalent
        direction_angle %= 360
        
        # Calculate bin size
        bin_size = 360 / num_bins
        
        # Determine bin index
        bin_index = int(direction_angle // bin_size)
        
        # Ensure bin index is within range [0, num_bins - 1]
        bin_index = max(min(bin_index, num_bins - 1), 0)
        
        return bin_index


    def discretize_distance(self,distance, min_distance, max_distance, num_bins):
        # Calculate bin size
        bin_size = (max_distance - min_distance) / num_bins
        
        # Determine bin index
        bin_index = int((distance - min_distance) // bin_size)
        
        # Ensure bin index is within range [0, num_bins - 1]
        bin_index = max(min(bin_index, num_bins - 1), 0)
        
        return bin_index



            

if __name__ == '__main__':
    game = PuzzleroomAI()
    
    # game loop
    while True:
        game_over = game.play_step()
        
        if game_over == True:
            break
        
        
    pygame.quit()