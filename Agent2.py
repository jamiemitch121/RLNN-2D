import torch
import random
import numpy as np
from collections import deque
from Game2 import PuzzleroomAI
from pprint import pprint
from model import Linear_QNet, QTrainer
from helper2 import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.00001

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 #how random the program is
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY) # removes from memory if max mem is reached
        self.model = Linear_QNet(11,256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)


    def get_state(self,game):
       

        state = [
        
        #circle locations
        game.discretize_coordinate(game.circlex,0,1000,10),
        game.discretize_coordinate(game.circley,0,500,5),
        game.discretize_coordinate(game.puzzlecx,0,1000,10),
        game.discretize_coordinate(game.puzzlecy,0,500,5),
        #game.boxx,
        #game.boxy,
        game.discretize_distance(game.dist,0,5100,10),



        #player to puzzle circle direction
        game.circlex < game.puzzlecx,
        game.circlex > game.puzzlecx,
        game.circley < game.puzzlecy,
        game.circley > game.puzzlecy,

        
        #puzzle circle to target
        #game.boxx < game.puzzlecx,
        #game.boxx > game.puzzlecx,
        #game.boxy < game.puzzlecy,
        #game.boxy > game.puzzlecy,

        #player angle
        game.discretize_direction(game.lowervec,8),
        #whether ball is colliding or not
        game.ballcol
        ]
        print(state)
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        self.losses = self.trainer.getloss()
        #for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 200 - self.n_games
        if self.epsilon < 0:
            self.epsilon = 2
        final_move = [0,0,0]
        if random.randint(0, 10) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        return final_move


def train():
    plot_times = []
    plot_mean_time = []
    total_time =0
    plot_rewards = []
    plot_mean_reward = []
    total_reward = 0
    plot_losses = []
    total_loss = 0
    plot_mean_loss =[]
    agent = Agent()
    game = PuzzleroomAI()

    while True:
        # old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, time = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()
            

            print('Game', agent.n_games, 'Time', time,'Reward', reward)
            loss = agent.losses
            plot_losses.append(loss)
            plot_times.append(time)
            total_loss += loss
            mean_loss = total_loss / agent.n_games
            plot_mean_loss.append(mean_loss)
            total_time += time
            mean_time = total_time / agent.n_games
            plot_mean_time.append(mean_time)
            

            plot_rewards.append(reward)
            total_reward += reward
            mean_reward = total_reward / agent.n_games
            plot_mean_reward.append(mean_reward)
            plot(plot_times, plot_mean_time,plot_rewards, plot_mean_reward, plot_losses,plot_mean_loss)


if __name__ == '__main__':
    train()