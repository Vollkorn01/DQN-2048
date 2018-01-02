# -*- coding: utf-8 -*-
import random
#import gym
import numpy as np
import random
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

#start in folder 2048-rl
from py_2048_rl.game.game import Game

EPISODES = 1000


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size # in our case 4*4*12
        self.action_size = action_size # ino our case 4 (up, down, right, left)
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.choice(game.available_actions())
        act_values = self.model.predict(state)
        #print(act_values[0])
        if np.argmax(act_values[0]) in game.available_actions():
            return np.argmax(act_values[0])
        #TODO: select 2nd highest value, if available action not available etc..
        else:
            return random.choice(game.available_actions())
    
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


if __name__ == "__main__":
    game = Game()
    #env = gym.make('CartPole-v1')
    state_size = 16
    #state_size = env.observation_space.shape[0] # in our case 4*4
    action_size = 4
    #action_size = env.action_space.n #in our case 4
    agent = DQNAgent(state_size, action_size)
    # agent.load("./save/cartpole-dqn.h5")
    done = False
    batch_size = 32

    for e in range(EPISODES):
        game.new_game()
        state = game.state()
        state = np.reshape(state, [1, state_size])
        #state = env.reset()
        while not game.game_over():
            #action = random.choice(game.available_actions()) #replace with epsilon greedy strategy
            #env.render()
            action = agent.act(state)
            reward = game.do_action(action)
            next_state = game.state()
            actions_available = game.available_actions()
            #print(actions_available)
            if len(actions_available) == 0: #wrong! implement function available_actions here instead
                done = True
            else:
                done = False
            #next_state, reward, done, _ = env.step(action)
            #reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            #print(state)

            if done:
                print("no action available")
                states = game.state()
                states = np.reshape(state, [1, state_size])
                max_value = np.amax(states[0])
                print(max_value)
                break
            #game.print_state()
        print("episodes: " + str(e))
        
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
        # if e % 10 == 0:
        #     agent.save("./save/cartpole-dqn.h5")
