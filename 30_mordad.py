#Lets define Class: "Simple Bandit" with 4 parameteres: (1) initializer, (2) string, (3) pull, (4) update**

import numpy as np                                                            #for generating random numbers
#because we utilize the normal distrubution

class Bandit:   #Like a template for making a slot machine

  def __init__ (self, q, Q):    #Initializer
    self.q = q    #true average reward
    self.Q = Q    #starting guess for average reward
    self.n = 0    #n starts at 0 because you haven't pulled the slot machine yet


  def __str__(self):    #string representation
  #tells Python how to print bandit's info including:
  #q (true average), Q (estimated avergae) and n (number of pulls)
  #in a nice format
    return f"q:{self.q} | Q:{self.Q} | n:{self.n}"


  def pull(self):   #Pull method, gives you random reward based on
  #bandit's true average
    return np.random.normal(self.q, 1)  #creates a random number close
    #to q (the true average) with some variation (the reward can vary a bit)
    #plus or minus 1
    #e.g., the q is 1, so you might get rewards like 0.7, 0.3 or 1.2

  def update(self, r): #Update method
  #updates our guess about the avergae reward after we pull a slot machine
  #r stands for reward
    self.n += 1   #adds 1 to the count of pulls
    self.Q = self.Q + (1.0/self.n)*(r - self.Q)
    #this formula updates our guess to actual rewards we're getting

b = Bandit(q = 0.5, Q = 0)
  #this codeline creates a bandit with true average reward or q of 0.5
  #also creates a bandit with starting guess of 0
  #Error: this codeline must be outside of the class
  #print(b)

for i in range(100):   #starting the loop
  #this loop plays the slop machine for 10 times
    r = b.pull()  #pulls the slot machine and gets a random reward
    b.update(r)
    #updates the bandit's guess (Q) based on that reward
    print(b)
    #shows the bandit's current state

"""**Context of the multi_armed_bandit problem using an Epsilon_Greedy algorithm**

A loop was created to compare the generated random number to "epsilon",
if this number is less than epsilon the algorithm chooses Exploration (trying a random bandit),
else it chooses Exploitation (selecting a bandit with highest estimated reward).
"""

import numpy as np #Already imported that
from matplotlib import pyplot as plt #for plotting the reward history at the end of the code

#This code simulates a scenario where you choose between multiple "Bandits" (think of it as a slot machine)
#inorder to maximize rewards; balancing exploration (trying new options) and exploitation (choosing the best known option)


def multi_armed_bandit_epsilon_greedy (num_bandits, epsilon, num_steps):
  #define a function called "multi_armed_bandit_epsilon_greedy"
  #that implmenets epsilon_greedy algorithm for the multi-armed bandit problem
  #3 parameters are recieved: (1)number of bandits, (2) epsilon as a value between 0 and 1 that controls the exploration-exploitation trade-off
  #number of steps as the number of iterations (or pulls) to run the simulation

  bandits = [Bandit(np.random.normal(0,1), Q = 0) for i in range(num_bandits)]
  #A mean reward drawn from a Guassian distribution (0,1)
  #Q stands for the initial estimated reward

  reward_history = []
  #Initializes an empty list to store the rewards obtained at each step of the simulation

  for i in range(num_steps):    #start the Loop
    if np.random.uniform(0,1)<epsilon:
      #exploration
      a = np.random.randint(0, num_bandits-1)
    else:
      #exploitation
      a = np.argmax([b.Q for b in bandits])

    r = bandits[a].pull()
    #Calls a pull method() on selected bandit to generate a reward
    bandits[a].update(r)
    #Updates the selected bandit's estimated reward (Q) using the recieved reward

    reward_history.append(r)
  return reward_history
  #Notice to write "retutn" codeline in the Loop

#Define the parameters for the simulation
num_bandits = 10    #simulates 10 bandits
num_steps = 1000    #runs the simulation for 1000 steps
epsilon = 0.1   #set 10% chance for simulation

reward_history = multi_armed_bandit_epsilon_greedy(num_bandits, epsilon, num_steps)
#call the "multi_armed_bandit_epsilon_greedy" function with defined parameters
#stores the returned "reward_history"

plt.plot(reward_history)
plt.show()

#This code implements a multi-armed bandit simulation, comparing 2 strategies
#1: standard ε-greedy
#2: Optimistic Initialization
#for choosing actions to maximize rewards.

import numpy as np
import matplotlib.pyplot as plt   #inorder to visualize the results

class Bandit:   #this class represents one arm of the multi-armed bandit

    def __init__(self, q, Q):   #constructor initializes the bandit
        self.q = q    #true mean reward
        self.Q = Q    #estimated reward
        self.n = 0    #tracks the number of times the bandit has been pulled

    def pull(self):   #simulates pulling the bandit's arm
        return np.random.normal(self.q, 1)    #returning a reward
                                              #this reward sampled from a normal distribution and standard deviation 1

    def update(self, r):    #updates the estimated reward (Q) after recieving reward (r)
        self.n += 1   #increments the count of pulls
        self.Q = self.Q + (r - self.Q) / self.n   #updates estimated reward using incremental avergae formula

def multi_armed_bandit(num_bandits, epsilon, num_steps, Q_init):
  #defines the mutli_armed_bandit function to run the bandit algorithm
  #recieves 4 parameters:
  #(1) num_bandits: number of bandit arms
  #(2) epsilon: probability of exploration (ε in ε-greedy)
  #(3) num_steps: number of time steps to run the simulation
  #(4) Q_init: initial estimated reward for all bandits

    bandits = [Bandit(np.random.normal(0, 1), Q_init) for i in range(num_bandits)]
    #each bandit's true reward mean (q) is sampled from a normal distribution and its estimated reward (Q)
    #is initialized to Q_init

    reward_history = []
    #initializes the empty list
    #to store the rewards obtained at each step

    for i in range(num_steps):    #Loops over num_steps

        if np.random.uniform(0, 1) < epsilon:       #with probability of epsilon

            # exploration
            a = np.random.randint(0, num_bandits)   #the algorithm explores by randomly selecting a bandit

        else:                                       #with probability 1-epsilon
            # exploitation
            a = np.argmax([b.Q for b in bandits])   #the algorithm exploits by selecting the bandit with highest estimated reward (Q)
                                                    #using (np.argmax)

        r = bandits[a].pull()                       #pulls a selected bandit to get a reward
        bandits[a].update(r)                        #updates the selected bandit's estimated reward (Q)


        reward_history.append(r)                    #Appends the rewards to reward_history
    return reward_history                           #returns the list of rewards obtained from num_steps

                                                    #Setting Paramters
num_bandits = 10                                    #10 bandits arms
num_steps = 1000                                    #1000 times steps per simulation
epsilon = 0.1                                       #10% chance of exploration in ε-greedy
num_cases = 500                                     #number of simulations to average over
Q_init_optimistic = 5                               #Initial Q value for optimistic initialization

                                                    # Run for epsilon = 0.1, Q = 0 (standard epsilon-greedy)
print(f'running for epsilon = {epsilon} Q = 0')     #prints a message indicating the start of the standatd ε-greedy simulation with
                                                    #Q_init = 0
avg_reward_0 = np.zeros(num_steps)                  #initializes an array of zeros with length "num_steps"
                                                    #to store the average rewards across simulations
for i in range(num_cases):                          #runs "num_cases" simulations with Q_init
    reward_history = multi_armed_bandit(num_bandits,
                                        epsilon,
                                        num_steps,
                                        Q_init=0)
    avg_reward_0 += np.array(reward_history)        #each simulation returns a "reward_history"

avg_reward_0 /= num_cases                           #which is converted to a NumPy array and added to avg_reward_0
                                                    #divides the "avg_reward_0" by "num_cases"
                                                    #to compute the average reward at each time step

plt.plot(avg_reward_0, label=f'epsilon = {epsilon}, Q = 0')

####################OPTIMISTIC INITIALIZATION####################################
# Run for epsilon = 0.1, Q = 5 (optimistic initialization)

print(f'running for epsilon = {epsilon} Q = {Q_init_optimistic}')
avg_reward_5 = np.zeros(num_steps)
for i in range(num_cases):
    reward_history = multi_armed_bandit(num_bandits, epsilon, num_steps, Q_init=Q_init_optimistic)
    avg_reward_5 += np.array(reward_history)
avg_reward_5 /= num_cases
plt.plot(avg_reward_5, label=f'epsilon = {epsilon}, Q = {Q_init_optimistic}')

plt.legend()
plt.show()

"""This code compares 3 multi-armed bandit strategies:"""

import math

#this code compares 3 multi-armed bandit strategies
#(1) Epsilon-Greedy with Q = 0: with 10% random choice, tries to balance exploration and exploitation
#(2) Epsilon-Greedy with Q = 5: Same as above but with optimistic initial estimates to encourage early exploration
#(3) UCB:Selects actions based on a combination of estimated rewards and uncertainty controlled by c.

class Bandit:                                                                     #defines a "Bandit" class to represent a single bandit in the multi-armed bandit problem
    def __init__(self, mean, Q_init=0):                                           #constructor initializes each bandit with:
        self.mean = mean                                                          #true mean reward of the bandit
        self.Q = Q_init                                                           #estimated value of the bandit
        self.n = 0                                                                #number of times the beandit has been chosen, initialized to 0

    def pull(self):                                                               #simulates pulling the bandit's arm, returning a reward drawn from a normal distribution
        return np.random.normal(self.mean, 1)                                     #with the bandit's true mean and a standard deviation of 1

    def update(self, r):                                                          #updates the bandit's estimated value (Q) based on the recieved reward(r)
        self.n += 1                                                               #increments self.n by 1 to track the number of pulls
        self.Q = self.Q + (r - self.Q) / self.n                                   #updates self.Q using incremental average formula: Q = Q + (r - Q) / n which computes the running average of rewards
        self.Q += (r - self.Q) / self.n

def multi_armed_bandit_ucb(num_bandits, num_steps, c):                            #Initializing a ucb (Upper Confidence Bound) function with 3 inputs: (1) num_bandits: number of bandits, (2) num_steps: number of time_steps (pulls) and (3) c: parameter controlling the trade-off between exploration and exploitation
    bandits = [Bandit(np.random.normal(0, 1)) for _ in range(num_bandits)]        #creates a list of num_bandits objects, each with a true mean drawn from a standard normal distribution (N(0,1))
    reward_history = []                                                           #initializes an empty list "reward_history" to store rewards over time

    for i in range(num_steps):                                                    #Loops over num_steps:
        ucb_values = [b.Q + c * math.sqrt(math.log(i + 1) / (b.n + 1e-6)) for b in bandits]
                                                                                  #Computes UCB values for each bandit: Q + c * sqrt(log(t + 1) / (n + 1e - 6))
                                                                                  #b.Q : current estimated value of the bandit
                                                                                  #c * math.sqrt(math.log(i + 1) / (b.n + 1e - 6): Exploration term, favoring bandits with fewer pulls (b.n) or high uncertainty.
                                                                                  # 1e - 6 prevents division by zero.
        a = np.argmax(ucb_values)                                                 #selects the bandit with the highest USB value
        r = bandits[a].pull()                                                     #pulls the selected bandit to get reward (r)
        bandits[a].update(r)                                                      #updates the selected bandit's estimate with the reward

        reward_history.append(r)                                                  #appends the rewars to "reward_history"
    return reward_history                                                         #returns the list of rewards to "reward_history"

def multi_armed_bandit(num_bandits, epsilon, num_steps, Q_init):                  #defines a function implementing the epsilon-greedy algorithm
                                                                                  #epsilon: probability of exploration (random action selection)
                                                                                  #Q_init: initial value for each bandit's Q (allow optimistic initialization)

    bandits = [Bandit(np.random.normal(0, 1), Q_init) for _ in range(num_bandits)]
                                                                                  #creates "num_bandits" bandits with random means and initial Q value of Q_init

    reward_history = []                                                           #initializes an empty "reward_history" list

    for i in range(num_steps):                                                    #loops over num_steps
        if np.random.uniform(0, 1) < epsilon:                                     #with probability "epsilon" explores by randomly selecting a bandit
            # exploration
            a = np.random.randint(0, num_bandits)
        else:
            # exploitation
            a = np.argmax([b.Q for b in bandits])                                 #with probability "1-epsilon" exploits by choosing the bandit with the highest Q value

        r = bandits[a].pull()
        bandits[a].update(r)

        reward_history.append(r)
    return reward_history                                                         #returns the list of reward

# Setting Parameters
c = 1                                                                             #UCB exploitation parameter
num_bandits = 10                                                                  #10 bandits in each experiment
num_steps = 1000                                                                  #1000 pulls per experiment
epsilon = 0.1                                                                     #10% chance of exploration in Epsilon-Greedy
num_cases = 500                                                                   #number of runs to average results for statistical reliability
Q_init_optimistic = 5                                                             #optimistic initial Q value for one experiment

# Run for epsilon = 0.1, Q = 0 (standard epsilon-greedy)
print(f'running for epsilon = {epsilon}, Q = 0')
avg_reward_eps0 = np.zeros(num_steps)
for i in range(num_cases):
    reward_history = multi_armed_bandit(num_bandits, epsilon, num_steps, Q_init=0)
    avg_reward_eps0 += np.array(reward_history)
avg_reward_eps0 /= num_cases

plt.plot(avg_reward_eps0, label=f'epsilon = {epsilon}, Q = 0')

# Run for optimistic initialization: epsilon = 0.1, Q = 5
print(f'running for epsilon = {epsilon}, Q = {Q_init_optimistic}')
avg_reward_optim = np.zeros(num_steps)
for i in range(num_cases):
    reward_history = multi_armed_bandit(num_bandits, epsilon, num_steps, Q_init=Q_init_optimistic)
    avg_reward_optim += np.array(reward_history)
avg_reward_optim /= num_cases                                                     #similar process as above: averages reward over num_cases runs and plots the results
plt.plot(avg_reward_optim, label=f'epsilon = {epsilon}, Q = {Q_init_optimistic}')

# Run for UCB
print(f'running for UCB with c = {c}')
avg_reward_ucb = np.zeros(num_steps)
for i in range(num_cases):
    reward_history = multi_armed_bandit_ucb(num_bandits, num_steps, c)
    avg_reward_ucb += np.array(reward_history)
avg_reward_ucb /= num_cases
plt.plot(avg_reward_ucb, label='UCB')

plt.legend()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
import math

# GBandit class (unchanged)
class GBandit:
    def __init__(self, q, H):
        self.q = q
        self.H = H
        self.n = 0

    def __str__(self):
        return f"q:{self.q} | H:{self.H} | n:{self.n}"

    def pull(self):
        return np.random.normal(self.q, 1)

    def update(self, new_H):
        self.n += 1
        self.H = new_H

# simple_gbandit function (unchanged)
def simple_gbandit(q):
    return GBandit(q=q, H=0)

# softmax function (unchanged)
def softmax(values):
    exp = np.exp(values)
    sum_exp = np.sum(exp)
    pi = exp / sum_exp
    return pi

# Corrected multi_armed_bandit_gradient function
def multi_armed_bandit_gradient(num_bandits, num_steps, alpha):
    bandits = [simple_gbandit(np.random.normal(0, 1)) for _ in range(num_bandits)]
    reward_history = []
    mean_R = 0
    n = 0  # Track number of rewards for mean_R update

    for i in range(num_steps):
        H_vals = [b.H for b in bandits]
        pi = softmax(H_vals)
        a = np.random.choice(num_bandits, p=pi)
        r = bandits[a].pull()
        reward_history.append(r)

        # Update mean_R incrementally
        n += 1
        mean_R += (r - mean_R) / n  # Incremental average formula

        # Update H for each bandit
        for j, b in enumerate(bandits):
            if j == a:
                new_H = b.H + alpha * (r - mean_R) * (1 - pi[j])
            else:
                new_H = b.H - alpha * (r - mean_R) * pi[j]
            b.update(new_H)

    return reward_history

# Rest of your code (Bandit class, multi_armed_bandit, multi_armed_bandit_ucb, etc.) remains unchanged
# ...

# Setting Parameters (unchanged)
alpha = 0.1
c = 1
num_bandits = 10
num_steps = 1000
epsilon = 0.1
num_cases = 500
Q_init_optimistic = 5

# Run for epsilon = 0.1, Q = 0 (unchanged)
print(f'running for epsilon = {epsilon}, Q = 0')
avg_reward_eps0 = np.zeros(num_steps)
for i in range(num_cases):
    reward_history = multi_armed_bandit(num_bandits, epsilon, num_steps, Q_init=0)
    avg_reward_eps0 += np.array(reward_history)
avg_reward_eps0 /= num_cases
plt.plot(avg_reward_eps0, label=f'epsilon = {epsilon}, Q = 0')

# Run for optimistic initialization: epsilon = 0.1, Q = 5 (unchanged)
print(f'running for epsilon = {epsilon}, Q = {Q_init_optimistic}')
avg_reward_optim = np.zeros(num_steps)
for i in range(num_cases):
    reward_history = multi_armed_bandit(num_bandits, epsilon, num_steps, Q_init=Q_init_optimistic)
    avg_reward_optim += np.array(reward_history)
avg_reward_optim /= num_cases
plt.plot(avg_reward_optim, label=f'epsilon = {epsilon}, Q = {Q_init_optimistic}')

# Run for UCB (unchanged)
print(f'running for UCB with c = {c}')
avg_reward_ucb = np.zeros(num_steps)
for i in range(num_cases):
    reward_history = multi_armed_bandit_ucb(num_bandits, num_steps, c)
    avg_reward_ucb += np.array(reward_history)
avg_reward_ucb /= num_cases
plt.plot(avg_reward_ucb, label='UCB')

# Corrected Run for Gradient Based
print(f'running for gradient-based with alpha = {alpha}')
avg_reward_gradient = np.zeros(num_steps)  # Use a separate variable
for i in range(num_cases):
    reward_history = multi_armed_bandit_gradient(num_bandits, num_steps, alpha)  # Use alpha, not c
    avg_reward_gradient += np.array(reward_history)
avg_reward_gradient /= num_cases
plt.plot(avg_reward_gradient, label=f'gradient-based, alpha = {alpha}')

# Finalize plot
plt.legend()
plt.show()                                                 #returns the list of reward

# Setting Parameters
alpha = 0.1
c = 1                                                                             #UCB exploitation parameter
num_bandits = 10                                                                  #10 bandits in each experiment
num_steps = 1000                                                                  #1000 pulls per experiment
epsilon = 0.1                                                                     #10% chance of exploration in Epsilon-Greedy
num_cases = 500                                                                   #number of runs to average results for statistical reliability
Q_init_optimistic = 5                                                             #optimistic initial Q value for one experiment

# Run for epsilon = 0.1, Q = 0 (standard epsilon-greedy)
print(f'running for epsilon = {epsilon}, Q = 0')
avg_reward_eps0 = np.zeros(num_steps)
for i in range(num_cases):
    reward_history = multi_armed_bandit(num_bandits, epsilon, num_steps, Q_init=0)
    avg_reward_eps0 += np.array(reward_history)
avg_reward_eps0 /= num_cases

plt.plot(avg_reward_eps0, label=f'epsilon = {epsilon}, Q = 0')

# Run for optimistic initialization: epsilon = 0.1, Q = 5
print(f'running for epsilon = {epsilon}, Q = {Q_init_optimistic}')
avg_reward_optim = np.zeros(num_steps)
for i in range(num_cases):
    reward_history = multi_armed_bandit(num_bandits, epsilon, num_steps, Q_init=Q_init_optimistic)
    avg_reward_optim += np.array(reward_history)
avg_reward_optim /= num_cases                                                     #similar process as above: averages reward over num_cases runs and plots the results
plt.plot(avg_reward_optim, label=f'epsilon = {epsilon}, Q = {Q_init_optimistic}')

# Run for UCB
print(f'running for UCB with c = {c}')
avg_reward_ucb = np.zeros(num_steps)
for i in range(num_cases):
    reward_history = multi_armed_bandit_ucb(num_bandits, num_steps, c)
    avg_reward_ucb += np.array(reward_history)
avg_reward_ucb /= num_cases
plt.plot(avg_reward_ucb, label='UCB')


#Run for Gradient Based
print(f'running for gradient_based = {alpha}')
avg_reward_ucb = np.zeros(num_steps)
for i in range(num_cases):
    reward_history = multi_armed_bandit_gradient(num_bandits, num_steps, c)
    avg_reward_ucb += np.array(reward_history)
avg_reward_ucb /= num_cases
plt.plot(avg_reward_ucb, label='gradient_based')
plt.legend()
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Define the Bandit class
class Bandit:
    def __init__(self, q):
        self.q = q  # True mean reward
        self.n = 0  # Number of times pulled
        self.q_estimate = 0  # Estimated mean reward

    def pull(self):
        # Return a reward sampled from a normal distribution with mean q and std 1
        return np.random.normal(self.q, 1)

    def update(self, reward):
        # Update the estimated mean reward
        self.n += 1
        self.q_estimate += (reward - self.q_estimate) / self.n

# UCB algorithm
def multi_armed_bandit_ucb(num_bandits, num_steps, c):
    bandits = [Bandit(np.random.normal(0, 1)) for _ in range(num_bandits)]
    reward_history = []
    total_pulls = 0

    # Initialize by pulling each bandit once
    for i in range(num_bandits):
        reward = bandits[i].pull()
        bandits[i].update(reward)
        reward_history.append(reward)
        total_pulls += 1

    # Main loop for remaining steps
    for i in range(num_steps - num_bandits):
        # Calculate UCB scores for each bandit
        ucb_scores = [
            bandit.q_estimate + c * np.sqrt(np.log(total_pulls + 1) / (bandit.n + 1e-5))
            for bandit in bandits
        ]
        # Select bandit with highest UCB score
        bandit_idx = np.argmax(ucb_scores)
        reward = bandits[bandit_idx].pull()
        bandits[bandit_idx].update(reward)
        reward_history.append(reward)
        total_pulls += 1

        # Update true mean every 100 steps
        if (i + num_bandits) % 100 == 0:
            for b in bandits:
                b.q = np.random.normal(0, 1)

    return reward_history

# Parameters
num_bandits = 10
num_steps = 1000
num_cases = 500
c = 1

# Run UCB algorithm and average rewards
avg_reward = np.zeros(num_steps)
print('Running for UCB')
for i in range(num_cases):
    reward_history = multi_armed_bandit_ucb(num_bandits, num_steps, c)
    for j in range(num_steps):
        avg_reward[j] += reward_history[j]

# Average the rewards
for i in range(num_steps):
    avg_reward[i] /= num_cases

# Plot the results
plt.plot(avg_reward, label='UCB')
plt.xlabel('Steps')
plt.ylabel('Average Reward')
plt.legend()
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Define the Bandit class (for ucb1, assuming sample-average updates)
class Bandit:
    def __init__(self, q):
        self.q = q  # True mean reward
        self.n = 0  # Number of times pulled
        self.q_estimate = 0  # Estimated mean reward

    def pull(self):
        # Return a reward sampled from a normal distribution with mean q and std 1
        return np.random.normal(self.q, 1)

    def update(self, reward):
        # Update the estimated mean reward using sample-average
        self.n += 1
        self.q_estimate += (reward - self.q_estimate) / self.n

# Define the NSBandit class (for ucb2, with fixed step-size alpha)
class NSBandit:
    def __init__(self, q, Q, alpha):
        self.q = q  # True mean reward
        self.n = 0  # Number of times pulled
        self.Q = Q  # Estimated mean reward
        self.alpha = alpha

    def pull(self):
        # Return a reward sampled from a normal distribution with mean q and std 1
        return np.random.normal(self.q, 1)

    def update(self, reward):
        # Update the estimated mean reward with fixed step-size
        self.n += 1
        self.Q = self.Q + self.alpha * (reward - self.Q)

# Helper function to create NSBandit instance
def simple_nsbandit(q):
    return NSBandit(q=q, Q=0, alpha=0.1)

# UCB algorithm for Bandit (sample-average updates)
def multi_armed_bandit_ucb1(num_bandits, num_steps, c):
    bandits = [Bandit(np.random.normal(0, 1)) for _ in range(num_bandits)]
    reward_history = []
    total_pulls = 0

    # Initialize by pulling each bandit once
    for i in range(num_bandits):
        reward = bandits[i].pull()
        bandits[i].update(reward)
        reward_history.append(reward)
        total_pulls += 1

    # Main loop for remaining steps
    for i in range(num_steps - num_bandits):
        # Calculate UCB scores for each bandit
        ucb_scores = [
            bandit.q_estimate + c * np.sqrt(np.log(total_pulls + 1) / (bandit.n + 1e-5))
            for bandit in bandits
        ]
        # Select bandit with highest UCB score
        bandit_idx = np.argmax(ucb_scores)
        reward = bandits[bandit_idx].pull()
        bandits[bandit_idx].update(reward)
        reward_history.append(reward)
        total_pulls += 1

        # Update true mean every 100 steps
        if (i + num_bandits) % 100 == 0:
            for b in bandits:
                b.q = np.random.normal(0, 1)

    return reward_history

# UCB algorithm for NSBandit (fixed step-size updates)
def multi_armed_bandit_ucb2(num_bandits, num_steps, c):
    bandits = [simple_nsbandit(np.random.normal(0, 1)) for _ in range(num_bandits)]
    reward_history = []
    total_pulls = 0

    # Initialize by pulling each bandit once
    for i in range(num_bandits):
        reward = bandits[i].pull()
        bandits[i].update(reward)
        reward_history.append(reward)
        total_pulls += 1

    # Main loop for remaining steps
    for i in range(num_steps - num_bandits):
        # Calculate UCB scores for each bandit
        ucb_scores = [
            bandit.Q + c * np.sqrt(np.log(total_pulls + 1) / (bandit.n + 1e-5))
            for bandit in bandits
        ]
        # Select bandit with highest UCB score
        bandit_idx = np.argmax(ucb_scores)
        reward = bandits[bandit_idx].pull()
        bandits[bandit_idx].update(reward)
        reward_history.append(reward)
        total_pulls += 1

        # Update true mean every 100 steps
        if (i + num_bandits) % 100 == 0:
            for b in bandits:
                b.q = np.random.normal(0, 1)

    return reward_history

# Parameters
num_bandits = 10
num_steps = 1000
num_cases = 500
c = 1

# Run ucb1 (sample-average Bandit)
avg_reward_ucb1 = np.zeros(num_steps)
print('Running for UCB1')
for i in range(num_cases):
    reward_history = multi_armed_bandit_ucb1(num_bandits, num_steps, c)
    for j in range(num_steps):
        avg_reward_ucb1[j] += reward_history[j]

# Average the rewards
for i in range(num_steps):
    avg_reward_ucb1[i] /= num_cases

# Run ucb2 (NSBandit with fixed step-size)
avg_reward_ucb2 = np.zeros(num_steps)
print('Running for UCB2')
for i in range(num_cases):
    reward_history = multi_armed_bandit_ucb2(num_bandits, num_steps, c)
    for j in range(num_steps):
        avg_reward_ucb2[j] += reward_history[j]

# Average the rewards
for i in range(num_steps):
    avg_reward_ucb2[i] /= num_cases

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(avg_reward_ucb1, label='UCB1 (Sample-Average)', color='red')
plt.plot(avg_reward_ucb2, label='UCB2 (Fixed Step-Size)', color='black')
plt.xlabel('Steps')
plt.ylabel('Average Reward')
plt.title('UCB1 vs UCB2 Performance')
plt.legend()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
alpha = 0.5
# Define the Bandit class (for ucb1, assuming sample-average updates)
class Bandit:
    def __init__(self, q):
        self.q = q  # True mean reward
        self.n = 0  # Number of times pulled
        self.q_estimate = 0  # Estimated mean reward

    def pull(self):
        # Return a reward sampled from a normal distribution with mean q and std 1
        return np.random.normal(self.q, 1)

    def update(self, reward):
        # Update the estimated mean reward using sample-average
        self.n += 1
        self.q_estimate += (reward - self.q_estimate) / self.n

# Define the NSBandit class (for ucb2, with fixed step-size alpha)
class NSBandit:
    def __init__(self, q, Q, alpha):
        self.q = q  # True mean reward
        self.n = 0  # Number of times pulled
        self.Q = Q  # Estimated mean reward
        self.alpha = alpha

    def pull(self):
        # Return a reward sampled from a normal distribution with mean q and std 1
        return np.random.normal(self.q, 1)

    def update(self, reward):
        # Update the estimated mean reward with fixed step-size
        self.n += 1
        self.Q = self.Q + self.alpha * (reward - self.Q)

# Helper function to create NSBandit instance
def simple_nsbandit(q):
    return NSBandit(q=q, Q=0, alpha=0.1)

# UCB algorithm for Bandit (sample-average updates)
def multi_armed_bandit_ucb1(num_bandits, num_steps, c):
    bandits = [Bandit(np.random.normal(0, 1)) for _ in range(num_bandits)]
    reward_history = []
    total_pulls = 0

    # Initialize by pulling each bandit once
    for i in range(num_bandits):
        reward = bandits[i].pull()
        bandits[i].update(reward)
        reward_history.append(reward)
        total_pulls += 1

    # Main loop for remaining steps
    for i in range(num_steps - num_bandits):
        # Calculate UCB scores for each bandit
        ucb_scores = [
            bandit.q_estimate + c * np.sqrt(np.log(total_pulls + 1) / (bandit.n + 1e-5))
            for bandit in bandits
        ]
        # Select bandit with highest UCB score
        bandit_idx = np.argmax(ucb_scores)
        reward = bandits[bandit_idx].pull()
        bandits[bandit_idx].update(reward)
        reward_history.append(reward)
        total_pulls += 1

        # Update true mean every 100 steps
        if (i + num_bandits) % 100 == 0:
            for b in bandits:
                b.q = np.random.normal(0, 1)

    return reward_history

# UCB algorithm for NSBandit (fixed step-size updates)
def multi_armed_bandit_ucb2(num_bandits, num_steps, c):
    bandits = [simple_nsbandit(np.random.normal(0, 1)) for _ in range(num_bandits)]
    reward_history = []
    total_pulls = 0

    # Initialize by pulling each bandit once
    for i in range(num_bandits):
        reward = bandits[i].pull()
        bandits[i].update(reward)
        reward_history.append(reward)
        total_pulls += 1

    # Main loop for remaining steps
    for i in range(num_steps - num_bandits):
        # Calculate UCB scores for each bandit
        ucb_scores = [
            bandit.Q + c * np.sqrt(np.log(total_pulls + 1) / (bandit.n + 1e-5))
            for bandit in bandits
        ]
        # Select bandit with highest UCB score
        bandit_idx = np.argmax(ucb_scores)
        reward = bandits[bandit_idx].pull()
        bandits[bandit_idx].update(reward)
        reward_history.append(reward)
        total_pulls += 1

        # Update true mean every 100 steps
        if (i + num_bandits) % 100 == 0:
            for b in bandits:
                b.q = np.random.normal(0, 1)

    return reward_history

# Parameters
num_bandits = 10
num_steps = 1000
num_cases = 500
c = 1

# Run ucb1 (sample-average Bandit)
avg_reward_ucb1 = np.zeros(num_steps)
print('Running for UCB1')
for i in range(num_cases):
    reward_history = multi_armed_bandit_ucb1(num_bandits, num_steps, c)
    for j in range(num_steps):
        avg_reward_ucb1[j] += reward_history[j]

# Average the rewards
for i in range(num_steps):
    avg_reward_ucb1[i] /= num_cases

# Run ucb2 (NSBandit with fixed step-size)
avg_reward_ucb2 = np.zeros(num_steps)
print('Running for UCB2')
for i in range(num_cases):
    reward_history = multi_armed_bandit_ucb2(num_bandits, num_steps, c)
    for j in range(num_steps):
        avg_reward_ucb2[j] += reward_history[j]

# Average the rewards
for i in range(num_steps):
    avg_reward_ucb2[i] /= num_cases

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(avg_reward_ucb1, label='UCB1 (Sample-Average)', color='red')
plt.plot(avg_reward_ucb2, label='UCB2 (Fixed Step-Size)', color='black')
plt.xlabel('Steps')
plt.ylabel('Average Reward')
plt.title('UCB1 vs UCB2 Performance')
plt.legend()
plt.show()

def negative_grid(step_cost=-0.1):  # Add step_cost as a parameter with a default value
    g = GridWorld(3, 4, (2,0))
    rewards = {
        (0,3): 1,
        (1,3): -1
    }

    # Assign step_cost to all non-terminal states
    for i in range(g.rows):
        for j in range(g.cols):
            if (i,j) not in rewards:
                rewards[(i,j)] = step_cost  # Correct dictionary assignment syntax

    actions = {
        (0,0): ('D', 'R'),
        (0,1): ('L', 'R'),
        (0,2): ('L', 'D', 'R'),
        (1,0): ('U', 'D'),
        (1,2): ('U', 'D', 'R'),
        (2,0): ('U', 'R'),
        (2,1): ('L', 'R'),
        (2,2): ('L', 'R', 'U'),
        (2,3): ('L', 'U')
    }

    probs = {
        ((2,0), 'U'): {(1,0): 1.0},
        ((2,0), 'D'): {(2,0): 1.0},
        ((2,0), 'L'): {(2,0): 1.0},
        ((2,0), 'R'): {(2,1): 1.0},
        ((1,0), 'U'): {(0,0): 1.0},
        ((1,0), 'D'): {(2,0): 1.0},
        ((1,0), 'L'): {(1,0): 1.0},
        ((1,0), 'R'): {(1,0): 1.0},
        ((0,0), 'U'): {(0,0): 1.0},
        ((0,0), 'D'): {(1,0): 1.0},
        ((0,0), 'L'): {(0,0): 1.0},
        ((0,0), 'R'): {(0,1): 1.0},
        ((0,1), 'U'): {(0,1): 1.0},
        ((0,1), 'D'): {(0,1): 1.0},
        ((0,1), 'L'): {(0,0): 1.0},
        ((0,1), 'R'): {(0,2): 1.0},
        ((0,2), 'U'): {(0,2): 1.0},
        ((0,2), 'D'): {(1,2): 1.0},
        ((0,2), 'L'): {(0,1): 1.0},
        ((0,2), 'R'): {(0,3): 1.0},
        ((1,2), 'U'): {(1,2): 1.0},
        ((1,2), 'D'): {(2,2): 1.0},
        ((1,2), 'L'): {(1,1): 1.0},
        ((1,2), 'R'): {(1,3): 1.0},
        ((2,2), 'U'): {(2,2): 1.0},
        ((2,2), 'D'): {(2,2): 1.0},
        ((2,2), 'L'): {(2,1): 1.0},
        ((2,2), 'R'): {(2,3): 1.0},
        ((2,3), 'U'): {(1,3): 1.0},
        ((2,3), 'D'): {(2,3): 1.0},
        ((2,3), 'L'): {(2,2): 1.0},
        ((2,3), 'R'): {(2,3): 1.0},
        ((1,1), 'U'): {(0,1): 1.0},
        ((1,1), 'D'): {(2,1): 1.0},
        ((1,1), 'L'): {(1,0): 1.0},
        ((1,1), 'R'): {(1,2): 1.0}
    }

    g.set(rewards, actions, probs)
    return g

import numpy as np
import matplotlib.pyplot as plt
alpha = 0.01
# Define the Bandit class (for ucb1, assuming sample-average updates)
class Bandit:
    def __init__(self, q):
        self.q = q  # True mean reward
        self.n = 0  # Number of times pulled
        self.q_estimate = 0  # Estimated mean reward

    def pull(self):
        # Return a reward sampled from a normal distribution with mean q and std 1
        return np.random.normal(self.q, 1)

    def update(self, reward):
        # Update the estimated mean reward using sample-average
        self.n += 1
        self.q_estimate += (reward - self.q_estimate) / self.n

# Define the NSBandit class (for ucb2, with fixed step-size alpha)
class NSBandit:
    def __init__(self, q, Q, alpha):
        self.q = q  # True mean reward
        self.n = 0  # Number of times pulled
        self.Q = Q  # Estimated mean reward
        self.alpha = alpha

    def pull(self):
        # Return a reward sampled from a normal distribution with mean q and std 1
        return np.random.normal(self.q, 1)

    def update(self, reward):
        # Update the estimated mean reward with fixed step-size
        self.n += 1
        self.Q = self.Q + self.alpha * (reward - self.Q)

# Helper function to create NSBandit instance
def simple_nsbandit(q):
    return NSBandit(q=q, Q=0, alpha=0.1)

# UCB algorithm for Bandit (sample-average updates)
def multi_armed_bandit_ucb1(num_bandits, num_steps, c):
    bandits = [Bandit(np.random.normal(0, 1)) for _ in range(num_bandits)]
    reward_history = []
    total_pulls = 0

    # Initialize by pulling each bandit once
    for i in range(num_bandits):
        reward = bandits[i].pull()
        bandits[i].update(reward)
        reward_history.append(reward)
        total_pulls += 1

    # Main loop for remaining steps
    for i in range(num_steps - num_bandits):
        # Calculate UCB scores for each bandit
        ucb_scores = [
            bandit.q_estimate + c * np.sqrt(np.log(total_pulls + 1) / (bandit.n + 1e-5))
            for bandit in bandits
        ]
        # Select bandit with highest UCB score
        bandit_idx = np.argmax(ucb_scores)
        reward = bandits[bandit_idx].pull()
        bandits[bandit_idx].update(reward)
        reward_history.append(reward)
        total_pulls += 1

        # Update true mean every 100 steps
        if (i + num_bandits) % 100 == 0:
            for b in bandits:
                b.q = np.random.normal(0, 1)

    return reward_history

# UCB algorithm for NSBandit (fixed step-size updates)
def multi_armed_bandit_ucb2(num_bandits, num_steps, c):
    bandits = [simple_nsbandit(np.random.normal(0, 1)) for _ in range(num_bandits)]
    reward_history = []
    total_pulls = 0

    # Initialize by pulling each bandit once
    for i in range(num_bandits):
        reward = bandits[i].pull()
        bandits[i].update(reward)
        reward_history.append(reward)
        total_pulls += 1

    # Main loop for remaining steps
    for i in range(num_steps - num_bandits):
        # Calculate UCB scores for each bandit
        ucb_scores = [
            bandit.Q + c * np.sqrt(np.log(total_pulls + 1) / (bandit.n + 1e-5))
            for bandit in bandits
        ]
        # Select bandit with highest UCB score
        bandit_idx = np.argmax(ucb_scores)
        reward = bandits[bandit_idx].pull()
        bandits[bandit_idx].update(reward)
        reward_history.append(reward)
        total_pulls += 1

        # Update true mean every 100 steps
        if (i + num_bandits) % 100 == 0:
            for b in bandits:
                b.q = np.random.normal(0, 1)

    return reward_history

# Parameters
num_bandits = 10
num_steps = 1000
num_cases = 500
c = 1

# Run ucb1 (sample-average Bandit)
avg_reward_ucb1 = np.zeros(num_steps)
print('Running for UCB1')
for i in range(num_cases):
    reward_history = multi_armed_bandit_ucb1(num_bandits, num_steps, c)
    for j in range(num_steps):
        avg_reward_ucb1[j] += reward_history[j]

# Average the rewards
for i in range(num_steps):
    avg_reward_ucb1[i] /= num_cases

# Run ucb2 (NSBandit with fixed step-size)
avg_reward_ucb2 = np.zeros(num_steps)
print('Running for UCB2')
for i in range(num_cases):
    reward_history = multi_armed_bandit_ucb2(num_bandits, num_steps, c)
    for j in range(num_steps):
        avg_reward_ucb2[j] += reward_history[j]

# Average the rewards
for i in range(num_steps):
    avg_reward_ucb2[i] /= num_cases

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(avg_reward_ucb1, label='UCB1 (Sample-Average)', color='red')
plt.plot(avg_reward_ucb2, label='UCB2 (Fixed Step-Size)', color='black')
plt.xlabel('Steps')
plt.ylabel('Average Reward')
plt.title('UCB1 vs UCB2 Performance')
plt.legend()
plt.show()

"""**Implementing a Simple Grid World**"""

import numpy as np
ACTION_SPACE = ('U', 'D', 'L', 'R')

class GridWorld:
  def __init__(self, rows, cols, start):
    self.rows = rows
    self.cols = cols
    self.i = start[0]
    self.j = start[1]
    self.start = start


  def set(self, rewards, actions, probs):
    # reward: a dictionary of {(r,c): r} ==> {(0,3):1, ...}
    # actions: a dictionaey of {(r,c): [actions]} ==> {(0,0): ['R','D'], ...}
    #probs: a dictionary of {((r,c), a): (r', c'): p} ==> {((0,0). 'R'): {(0,1): 0.5, (1,0): 0.5}}

    self.rewards = rewards
    self.actions = actions
    self.probs = probs

  def set_state(self, s):
    self.i = s[0]
    self.j = s[1]

  def current_state(self):
    return(self.i, self.j)

  def is_terminal(self, s):
    return s not in self.actions

  def move(self, s):
    s = (self.i, self.j)
    next_state_probs = self.probs[(s,a)]
    next_states = list(next_state_probs.keys())
    next_probs = list(next_state_probs.values())

    idx = np.random.choice(len(next_states), p = next_probs)

    s2 = next_states[idx]

    #update the current state
    self.i, self.j = s2

    #return a reward if any
    return self.rewards.get(s2, 0)

  def game_over(self):
    return (self.i, self.j) not in self.actions

  def all_states(self):
    return set(self.actions.keys()  | self.rewards.keys())

  def reset(self):
    self.i, self.j = self.start
    return self.start



def simple_grid():
  g = GridWorld(3, 4, (2,0))
  rewards = {
      (0,3): 1,
      (1,3): -1
  }

  actions = {
      (0,0): ('D', 'R'),
      (0,1): ('L', 'R'),
      (0,2): ('L', 'D', 'R'),
      (1,0): ('U', 'D'),
      (1,2): ('U', 'D', 'R'),
      (2,0): ('U', 'R'),
      (2,1): ('L', 'R'),
      (2,2): ('L', 'R', 'U'),
      (2,3): ('L', 'U')
  }

  probs = {
      ((2,0), 'U'): {(1,0): 1.0},
      ((2,0), 'D'): {(2,0): 1.0},
      ((2,0), 'L'): {(2,0): 1.0},
      ((2,0), 'R'): {(2,1): 1.0},
      ((1,0), 'U'): {(0,0): 1.0},
      ((1,0), 'D'): {(2,0): 1.0},
      ((1,0), 'L'): {(1,0): 1.0},
      ((1,0), 'R'): {(1,0): 1.0},
      ((0,0), 'U'): {(0,0): 1.0},
      ((0,0), 'D'): {(1,0): 1.0},
      ((0,0), 'L'): {(0,0): 1.0},
      ((0,0), 'R'): {(0,1): 1.0},
      ((0,1), 'U'): {(0,1): 1.0},
      ((0,1), 'D'): {(0,1): 1.0},
      ((0,1), 'L'): {(0,0): 1.0},
      ((0,1), 'R'): {(0,2): 1.0},
      ((0,2), 'U'): {(0,2): 1.0},
      ((0,2), 'D'): {(1,2): 1.0},
      ((0,2), 'L'): {(0,1): 1.0},
      ((0,2), 'R'): {(0,3): 1.0},
      ((1,2), 'U'): {(1,2): 1.0},
      ((1,2), 'D'): {(2,2): 1.0},
      ((1,2), 'L'): {(1,1): 1.0},
      ((1,2), 'R'): {(1,3): 1.0},
      ((2,2), 'U'): {(2,2): 1.0},
      ((2,2), 'D'): {(2,2): 1.0},
      ((2,2), 'L'): {(2,1): 1.0},
      ((2,2), 'R'): {(2,3): 1.0},
      ((2,3), 'U'): {(1,3): 1.0},
      ((2,3), 'D'): {(2,3): 1.0},
      ((2,3), 'L'): {(2,2): 1.0},
      ((2,3), 'R'): {(2,3): 1.0},
      ((1,1), 'U'): {(0,1): 1.0},
      ((1,1), 'D'): {(2,1): 1.0},
      ((1,1), 'L'): {(1,0): 1.0},
      ((1,1), 'R'): {(1,2): 1.0}
  }

  g.set(rewards, actions, probs)
  return g



def windy_grid():
  g = GridWorld(3, 4, (2,0))
  rewards = {
      (0,3): 1,
      (1,3): -1
  }

  actions = {
      (0,0): ('D', 'R'),
      (0,1): ('L', 'R'),
      (0,2): ('L', 'D', 'R'),
      (1,0): ('U', 'D'),
      (1,2): ('U', 'D', 'R'),
      (2,0): ('U', 'R'),
      (2,1): ('L', 'R'),
      (2,2): ('L', 'R', 'U'),
      (2,3): ('L', 'U')
  }

  probs = {
      ((2,0), 'U'): {(1,0): 1.0},
      ((2,0), 'D'): {(2,0): 1.0},
      ((2,0), 'L'): {(2,0): 1.0},
      ((2,0), 'R'): {(2,1): 1.0},
      ((1,0), 'U'): {(0,0): 1.0},
      ((1,0), 'D'): {(2,0): 1.0},
      ((1,0), 'L'): {(1,0): 1.0},
      ((1,0), 'R'): {(1,0): 1.0},
      ((0,0), 'U'): {(0,0): 1.0},
      ((0,0), 'D'): {(1,0): 1.0},
      ((0,0), 'L'): {(0,0): 1.0},
      ((0,0), 'R'): {(0,1): 1.0},
      ((0,1), 'U'): {(0,1): 1.0},
      ((0,1), 'D'): {(0,1): 1.0},
      ((0,1), 'L'): {(0,0): 1.0},
      ((0,1), 'R'): {(0,2): 1.0},
      ((0,2), 'U'): {(0,2): 1.0},
      ((0,2), 'D'): {(1,2): 1.0},
      ((0,2), 'L'): {(0,1): 1.0},
      ((0,2), 'R'): {(0,3): 1.0},
      ((1,2), 'U'): {(0,2): 0.5, (1,3):0.5 },
      ((1,2), 'D'): {(2,2): 1.0},
      ((1,2), 'L'): {(1,1): 1.0},
      ((1,2), 'R'): {(1,3): 1.0},
      ((2,2), 'U'): {(2,2): 1.0},
      ((2,2), 'D'): {(2,2): 1.0},
      ((2,2), 'L'): {(2,1): 1.0},
      ((2,2), 'R'): {(2,3): 1.0},
      ((2,3), 'U'): {(1,3): 1.0},
      ((2,3), 'D'): {(2,3): 1.0},
      ((2,3), 'L'): {(2,2): 1.0},
      ((2,3), 'R'): {(2,3): 1.0},
      ((1,1), 'U'): {(0,1): 1.0},
      ((1,1), 'D'): {(2,1): 1.0},
      ((1,1), 'L'): {(1,0): 1.0},
      ((1,1), 'R'): {(1,2): 1.0}
  }

  g.set(rewards, actions, probs)
  return g



def negative_grid(step_cost):
    g = GridWorld(3, 4, (2,0))
    rewards = {
        (0,3): 1,
        (1,3): -1
    }

    for i in range(g.rows):
        for j in range(g.cols):
            if (i,j) not in rewards:
                rewards[(i,j)] = step_cost

    actions = {
        (0,0): ('D', 'R'),
        (0,1): ('L', 'R'),
        (0,2): ('L', 'D', 'R'),
        (1,0): ('U', 'D'),
        (1,2): ('U', 'D', 'R'),
        (2,0): ('U', 'R'),
        (2,1): ('L', 'R'),
        (2,2): ('L', 'R', 'U'),
        (2,3): ('L', 'U')
    }

    probs = {
        ((2,0), 'U'): {(1,0): 1.0},
        ((2,0), 'D'): {(2,0): 1.0},
        ((2,0), 'L'): {(2,0): 1.0},
        ((2,0), 'R'): {(2,1): 1.0},
        ((1,0), 'U'): {(0,0): 1.0},
        ((1,0), 'D'): {(2,0): 1.0},
        ((1,0), 'L'): {(1,0): 1.0},
        ((1,0), 'R'): {(1,0): 1.0},
        ((0,0), 'U'): {(0,0): 1.0},
        ((0,0), 'D'): {(1,0): 1.0},
        ((0,0), 'L'): {(0,0): 1.0},
        ((0,0), 'R'): {(0,1): 1.0},
        ((0,1), 'U'): {(0,1): 1.0},
        ((0,1), 'D'): {(0,1): 1.0},
        ((0,1), 'L'): {(0,0): 1.0},
        ((0,1), 'R'): {(0,2): 1.0},
        ((0,2), 'U'): {(0,2): 1.0},
        ((0,2), 'D'): {(1,2): 1.0},
        ((0,2), 'L'): {(0,1): 1.0},
        ((0,2), 'R'): {(0,3): 1.0},
        ((1,2), 'U'): {(1,2): 1.0},
        ((1,2), 'D'): {(2,2): 1.0},
        ((1,2), 'L'): {(1,1): 1.0},
        ((1,2), 'R'): {(1,3): 1.0},
        ((2,2), 'U'): {(2,2): 1.0},
        ((2,2), 'D'): {(2,2): 1.0},
        ((2,2), 'L'): {(2,1): 1.0},
        ((2,2), 'R'): {(2,3): 1.0},
        ((2,3), 'U'): {(1,3): 1.0},
        ((2,3), 'D'): {(2,3): 1.0},
        ((2,3), 'L'): {(2,2): 1.0},
        ((2,3), 'R'): {(2,3): 1.0},
        ((1,1), 'U'): {(0,1): 1.0},
        ((1,1), 'D'): {(2,1): 1.0},
        ((1,1), 'L'): {(1,0): 1.0},
        ((1,1), 'R'): {(1,2): 1.0}
    }

    g.set(rewards, actions, probs)
    return g

#g = negative_grid(step_cost=-0.1)
#print(g.rewards)

import numpy as np

def iterative_policy_evaluation(policy, grid, threshold = 1.0e-3, gamma = 0.9):
    #gathering information from the grid
    transition_probs = {}
    rewards = {}
    for (s,a), v in grid.probs.items():
      for s2, p in v.items():
        transition_probs[(s,a,s2)] = p
        rewards[(s,a,s2)] = grid.rewards.get(s2,0)

    #Intialization
    V = {}
    for s in grid.all_states():
      V[s] = 0
    #repeat
    it = 0
    while True:
      delta = 0
      for s in grid.all_states():
        if not grid.is_terminal(s):
          old_v = V[s]
          new_v = 0
          for a in ACTION_SPACE:
            for s2 in grid.all_states():
              t = transition_probs.get((s,a,s2), 0)
              pi = policy[s].get(a,0)
              r = rewards.get((s,a,s2), 0)
              new_v += pi*t*(r+gamma*V[s2])
            V[s] = new_v
            delta = max(delta, np.abs(old_v - V[s]))

        it += 1
        if delta < threshold:
          return V

def print_values(V, g):
  for i in range(g.rows):
    print("----------------------"*3)
    for j in range(g.cols):
      s = (i,j)
      v = V.get(s,0)
      print("\t%.2f\t|"%v, end = "")
    print("")


policy = {
    (2, 0): {'U': 1.0},
    (1, 0): {'U': 1.0},
    (0, 0): {'R': 1.0},
    (0, 1): {'R': 1.0},
    (0, 2): {'R': 1.0},
    (1, 2): {'U': 1.0},
    (2, 1): {'R': 1.0},
    (2, 2): {'U': 1.0},
    (2, 3): {'L': 1.0},
}

grid = simple_grid()

V = iterative_policy_evaluation(policy, grid)

print_values(V, grid)

#create a random policy
def random_policy(grid):
  policy = {}
  for s in grid.actions.keys():
    random_action = np.random.choice(ACTION_SPACE)
    policy[s] = {random_action: 1.0}
  return policy

  #policy evaluation
  def iterative_policy_evaluation(policy, grid, threshold = 1.0e-3, gamma = 0.9):
    #gathering information from the grid
    transition_probs = {}
    rewards = {}
    for (s,a), v in grid.probs.items():
      for s2, p in v.items():
        transition_probs[(s,a,s2)] = p
        rewards[(s,a,s2)] = grid.rewards.get(s2,0)

def calculate_probs_and_rewards(grid):

    #Initialization
    def iterative_policy_evaluation(policy, grid, threshold = 1.0e-3, gamma = 0.9):
        transition_probs,

import random
from matplotlib import pyplot as plt

num_rolls = 10000
sample_means = []
rolls_sum = 0.0

for i in range(num_rolls):
    roll = random.randint(1,6)
    rolls_sum += roll
    sample_mean = rolls_sum / (i+1)
    sample_means.append(sample_mean)
expected_mean = [3.5]*num_rolls

plt.plot(range(1,num_rolls+1), sample_means, label = "Sample Means")
plt.plot(range(1,num_rolls+1), expected_mean, label = "Expected Mean", linestyle = "--")

plt.xlabel("Num of rolls")
plt.ylabel("Mean")

plt.legend()
plt.show()

import numpy as np

class Grid:
    def __init__(self, rows=3, cols=4):
        self.rows = rows
        self.cols = cols
        self.actions = {
            (0, 0): ['R', 'D'], (0, 1): ['L', 'R'], (0, 2): ['L', 'R', 'D'],
            (1, 0): ['U', 'D'], (1, 2): ['U', 'D', 'R'], (2, 0): ['U', 'R'],
            (2, 1): ['L', 'R'], (2, 2): ['L', 'R', 'U'], (2, 3): ['U', 'L']
        }
        self.rewards = {(0, 3): 1, (1, 3): -1}
        self.state = (2, 0)

    def set_state(self, state):
        self.state = state

    def current_state(self):
        return self.state

    def move(self, action):
        if self.game_over():
            return 0
        i, j = self.state
        next_state = self.state
        if action == 'U' and (i-1, j) in self.all_states() and (i-1, j) not in self.rewards:
            next_state = (i-1, j)
        elif action == 'D' and (i+1, j) in self.all_states() and (i+1, j) not in self.rewards:
            next_state = (i+1, j)
        elif action == 'L' and (i, j-1) in self.all_states() and (i, j-1) not in self.rewards:
            next_state = (i, j-1)
        elif action == 'R' and (i, j+1) in self.all_states() and (i, j+1) not in self.rewards:
            next_state = (i, j+1)
        self.state = next_state
        return self.rewards.get(self.state, 0)

    def game_over(self):
        return self.state in self.rewards

    def all_states(self):
        states = []
        for i in range(self.rows):
            for j in range(self.cols):
                states.append((i, j))
        return states

def play_game(grid, policy, max_step=20):
    start_states = list(grid.actions.keys())
    start_idx = np.random.choice(len(start_states))
    grid.set_state(start_states[start_idx])

    s = grid.current_state()
    states = [s]
    rewards = [0]
    step = 0

    while not grid.game_over():
        a = list(policy.get(s, {'None': 1.0}).keys())[0]
        if a == 'None':
            break
        r = grid.move(a)
        next_s = grid.current_state()
        states.append(next_s)
        rewards.append(r)
        step += 1
        if step >= max_step:
            break
        s = next_s
    return states, rewards

def monte_carlo_evaluation(policy, grid, gamma=0.9):
    V = {}
    returns = {}
    states = grid.all_states()
    for s in states:
        V[s] = 0
        if s in grid.actions:
            returns[s] = []

    for it in range(1000):
        states, rewards = play_game(grid, policy)
        G = 0
        T = len(states)
        for t in range(T-2, -1, -1):
            s = states[t]
            r = rewards[t+1]
            G = r + gamma * G
            if s not in states[:t]:
                if s in returns:
                    returns[s].append(G)
                    V[s] = np.mean(returns[s])
    return V

def print_values(V, g):
    for i in range(g.rows):
        print("---------------------" * 3)
        for j in range(g.cols):
            s = (i, j)
            v = V.get(s, 0)
            print("\t%.2f\t|" % v, end="")
        print("")

def print_policy(P, g):
    for i in range(g.rows):
        print("---------------------" * 3)
        for j in range(g.cols):
            a = P.get((i, j), {}).get('U', P.get((i, j), {}).get('D', P.get((i, j), {}).get('L', P.get((i, j), {}).get('R', ''))))
            print(f"\t{a}\t|", end="")
            print("")

policy = {
    (2, 0): {'U': 1.0},
    (1, 0): {'U': 1.0},
    (0, 0): {'R': 1.0},
    (0, 1): {'R': 1.0},
    (0, 2): {'R': 1.0},
    (1, 2): {'R': 1.0},
    (2, 1): {'R': 1.0},
    (2, 2): {'R': 1.0},
    (2, 3): {'U': 1.0},
}

grid = Grid()
V = monte_carlo_evaluation(policy, grid)
print_values(V, grid)
print("\nPolicy:")
print_policy(policy, grid)

def print_policy(P, g):
    for i in range(g.rows):
        print("------------------"*3)
        for j in range(g.cols):
            a = P.get((i, j), '')
            print(" %s |"%a, end = "")
        print("")

grid = simple_grid()
gamma = 0.9


policy = monte_carlo_estimation_es(grid, gamma)

print_policy(policy, grid)

import numpy as np

class Grid:
    def __init__(self, rows=3, cols=4):
        self.rows = rows
        self.cols = cols
        self.actions = {
            (0, 0): ['R', 'D'], (0, 1): ['L', 'R'], (0, 2): ['L', 'R', 'D'],
            (1, 0): ['U', 'D'], (1, 2): ['U', 'D', 'R'], (2, 0): ['U', 'R'],
            (2, 1): ['L', 'R'], (2, 2): ['L', 'R', 'U'], (2, 3): ['U', 'L']
        }
        self.rewards = {(0, 3): 1, (1, 3): -1}
        self.state = (2, 0)

    def set_state(self, state):
        self.state = state

    def current_state(self):
        return self.state

    def move(self, action):
        if self.game_over():
            return 0
        i, j = self.state
        next_state = self.state
        if action == 'U' and (i-1, j) in self.all_states() and (i-1, j) not in self.rewards:
            next_state = (i-1, j)
        elif action == 'D' and (i+1, j) in self.all_states() and (i+1, j) not in self.rewards:
            next_state = (i+1, j)
        elif action == 'L' and (i, j-1) in self.all_states() and (i, j-1) not in self.rewards:
            next_state = (i, j-1)
        elif action == 'R' and (i, j+1) in self.all_states() and (i, j+1) not in self.rewards:
            next_state = (i, j+1)
        self.state = next_state
        return self.rewards.get(self.state, 0)

    def game_over(self):
        return self.state in self.rewards

    def all_states(self):
        states = []
        for i in range(self.rows):
            for j in range(self.cols):
                states.append((i, j))
        return states

    def reset(self):
        self.state = (2, 0)  # Reset to initial state
        return self.state

def epsilon_greedy(grid, policy, s, eps=0.1):
    if s not in grid.actions:
        return None  # No action possible in terminal states
    p = np.random.random()
    if p < 1 - eps:
        # Choose the action from the policy (assumes policy[s] is a dict with probabilities)
        return list(policy[s].keys())[0]
    else:
        return np.random.choice(grid.actions[s])

def play_game(grid, policy, max_step=20):
    # Reset the game
    s = grid.reset()
    a = epsilon_greedy(grid, policy, s)

    states = [s]
    rewards = [0]
    actions = []
    step = 0

    while not grid.game_over() and step < max_step:
        r = grid.move(a)
        next_s = grid.current_state()

        rewards.append(r)
        states.append(next_s)
        actions.append(a)

        if grid.game_over():
            break

        a = epsilon_greedy(grid, policy, next_s)
        step += 1

    return states, rewards

def monte_carlo_estimation_no_es(policy, grid, gamma=0.9):
    V = {}
    returns = {}
    states = grid.all_states()
    for s in states:
        V[s] = 0
        if s in grid.actions:
            returns[s] = []

    for _ in range(1000):
        states, rewards = play_game(grid, policy)
        G = 0
        T = len(states)
        for t in range(T-1, -1, -1):
            s = states[t]
            r = rewards[t]
            G = r + gamma * G
            if s not in states[:t] and s in returns:
                returns[s].append(G)
                V[s] = np.mean(returns[s])
    return V

def print_values(V, g):
    for i in range(g.rows):
        print("---------------------" * 3)
        for j in range(g.cols):
            s = (i, j)
            v = V.get(s, 0)
            print("\t%.2f\t|" % v, end="")
        print("")

def print_policy(P, g):
    for i in range(g.rows):
        print("---------------------" * 3)
        for j in range(g.cols):
            s = (i, j)
            a = P.get(s, {}).get('U', P.get(s, {}).get('D', P.get(s, {}).get('L', P.get(s, {}).get('R', ''))))
            print(f"\t{a}\t|", end="")
        print("")

# Define the policy
policy = {
    (2, 0): {'U': 1.0},
    (1, 0): {'U': 1.0},
    (0, 0): {'R': 1.0},
    (0, 1): {'R': 1.0},
    (0, 2): {'R': 1.0},
    (1, 2): {'R': 1.0},
    (2, 1): {'R': 1.0},
    (2, 2): {'R': 1.0},
    (2, 3): {'U': 1.0},
}

# Run the Monte Carlo estimation
grid = Grid()
V = monte_carlo_estimation_no_es(policy, grid)
print("Values:")
print_values(V, grid)
print("\nPolicy:")
print_policy(policy, grid)

#SARSA Algorithm

def find_dict(dictionary):
  max_value = max(dictionary.values())
  max_keys = []

  for key, value in dictionary.items():
    if value == max_value:
         max_keys.append(key)
  random_max_key = np.random.choice(max_keys)

  return random_max_key, max_value

def epsilon_greedy(grid, Q, s, eps = 0.1):
    p = np. random.random()
    if p<1-eps:
        a = find_dict(Q[s])[0]
        return a
    else:
      return np.random.choice(grid.actions[s])

def sarsa(grid, gamma = 0.9, alpha = 0.1):
    #Initialize Q(s,a)
    Q = {}
    states = grid.all_states()
    for s in states:
        if s in grid.actions:
            Q[s] = {}
            for a in grid.actions[s]:
              Q[s][a] = 0
    for it in range(1000):
      s = grid.reset()
      a = epsilon_greedy(grid, Q, s)
      while not grid.game_over():
        r = grid.move(a)
        s2 = grid.current_state()
        if grid.is_terminal(s2):
          Q[s][a] = Q[s][a] + alpha*(r-Q[s][a])
          break

        a2 = epsilon_greedy(grid, Q, s2)
        Q[s][a] = Q[s][a] + alpha*(r+gamma*Q[s2][a2]-Q[s][a])
        s = s2
        a = a2

    #Find the policy from Q*
    policy = {}
    for s in grid.actions.keys():
        max_action, max_value = find_dict(Q[s])
        policy[s] = {max_action: 1.0}

    return policy


def print_policy(P, g):
    for i in range(g.rows):
      print("--------------------"*3)
      for j in range(g.cols):
          a = P.get((i, j), ' ')
          print("%s  |"%a, end = "")
      print("")


grid = simple_grid()
#try this line later:
#grid = negative_grid(-2)

policy = sarsa(grid)

print_policy(policy, grid)
