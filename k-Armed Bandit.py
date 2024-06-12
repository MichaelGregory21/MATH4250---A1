import random
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import seaborn as sns

'''
A Bandit is a slot machine which can give an agent a random reward sampled from a normal distrbution centered at the given mean
A Bandit comes equipped with a method that returns an estimated reward based on all previous rewards given equal weight
'''
class Bandit:
    # The standard deviation of a bandit
    _DEVIATION = 1

    '''
    Constructor for objects of type Bandit
    @param int mean The mean reward recieved from spinning this bandit. 0, by default
    @param float expected The initial estimate for the reward from this bandit. 0, by default
    @param lambda alpha The update rule when estimating the reward associated with a bandit. 1/n (uniform average), by default
    '''
    def __init__(self, mean=0, expected=0, alpha=lambda n: 0 if n==0 else 1/n):
        self._mean = mean
        self._prev_reward = 0
        self._expected = expected
        self._spins = 0
        self._update_rule = alpha

    '''
    Spin this bandit and track the score
    @return float A random score from a normal distribution with this bandit's mean
    '''
    def spin(self):
        self._spins += 1
        score = np.random.normal(loc=self._mean, scale=self._DEVIATION)
        self.adjust_estimate(score, self._update_rule)
        self._prev_reward = score
        return score

    '''
    Adjust the estimated score associated with this bandit
    @param float new_score The score to be considered in adjusting the expected reward from this bandit
    @param lambda The update rule associated with this estimate
    '''
    def adjust_estimate(self, new_score, alpha):
        self._expected += alpha(self._spins) * (self._prev_reward - self._expected)
        

    '''
    Accessor for the expected reward from spinning this bandit
    @return float The expected reward from spinning this bandit
    '''
    def estimate(self):
        return self._expected

    '''
    Plot a histogram of a set of sampled rewards from this bandit behind the theoretical, expected normal distribution
    @param str title The title for the plot
    @param str xlabel The label for the x-axis of the plot
    @param str ylabel The label for the y-axis of the plot
    @param int samples The number of samples you want to take
    '''
    def plot_rewards(self, title="Sampled Bandit Rewards", xlabel="Reward", ylabel="Density", samples=1000):
        rewards = [self.spin() for spin in range(samples)]
        plt.hist(rewards, bins=20, density=True, alpha=0.6, color='g')
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = (1 / (np.sqrt(2 * np.pi) * self._DEVIATION)) * np.exp(-((x - self._mean)**2 / (2 * self._DEVIATION**2)))
        plt.plot(x, p, 'k', linewidth=2)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)
        self.show_plot(title, xlabel, ylabel)

    '''
    Plot this bandits mean value with respect to time according to the given function
    @param lambda shift The function according to which the bandit mean will be altered. Identity, by default.
    @param int steps The number of steps you want to plot. 2000, by default.
    @param str label The label of this bandits plot. None, by default
    @param str color The color of the plot. 'r' (red), by default.
    '''
    def plot_bandit_mean(self, shift=lambda m: m, steps=2000, label=None, color='r'):
        values = [self._mean]
        for _ in range(steps):
            self._mean = shift(self._mean)
            values.append(self._mean)
        x_values = list(range(steps + 1))
        plt.plot(x_values, values, marker='o', markersize=2, linestyle='', color=color, label=label)
        plt.grid(True)

    '''
    Show the plot designed by this bandit
    @param str title The title of the plot
    @param str xaxis The label on the x axis of the plot
    @param str yaxis The label on the y axis of the plot
    @param str legend_loc The location of the legend of the plot. "upper left", by default.
    '''
    def show_plot(self, title, xaxis, yaxis, legend_loc="upper left"):
        plt.title(title)
        plt.xlabel(xaxis)
        plt.ylabel(yaxis)
        plt.legend(loc=legend_loc, fontsize=13)
        plt.show()


    '''
    Mutator for the mean of this bandit
    @param int mean The new mean you want to set for this bandit
    '''
    def setMean(self, mean):
        self._mean = mean

    '''
    Accessor for the mean of this bandit
    @return float The mean of this bandit
    '''
    def getMean(self):
        return self._mean
    
'''
An Environment is a collection of iterable bandits
'''
class Environment:

    '''
    Constructor for objects of type Environment
    @param float bandit_range_variance The variance of the distribution from which the bandit rewards' are sampled from the positive values
    @param int num_bandits The number of bandits to be in this environment
    @param float initial_estimate The initial expectation of the reward for every bandit in this environment. 0, by default
    @param lambda alpha The update rule when estimating the reward associated with a bandit. 1/n (uniform average), by default
    '''
    def __init__(self, bandit_range_variance=1, num_bandits=10, initial_estimate=0, alpha=lambda n: 0 if n==0 else 1/n):
        self._bandits = [Bandit(np.random.normal(scale=bandit_range_variance**2), expected=initial_estimate, alpha=alpha)
                         for i in range(num_bandits)]

    '''
    Plot a sampled reward distribution from each bandit and display all in a violin plot
    @param str title The title of the plot. "Sampled Reward Distribution for Each Bandit", by default
    @param str xlabel The x axis label of the plot. "Bandit", by default
    @param str ylabel The y axis label of the plot. "Reward", by default
    @param int samples The number of samples from each bandit. 1000, by default
    '''
    def plot_bandits(self, title="Sampled Reward Distribution for Each Bandit", xlabel="Bandit", ylabel="Reward", samples=1000, permute=False):
        data = [(bandit.getMean(), np.random.normal(bandit.getMean(), Bandit._DEVIATION, size=samples)) for bandit in self]
        if permute: random.shuffle(data)
        plt.figure(figsize=(8,6))
        violin_parts = plt.violinplot([i[1] for i in data], showmeans=True)
        for j in range(len(data)):
            x_coord = j + 1
            y_coord = data[j][0]
            plt.text(x_coord, y_coord, f'{y_coord:.2f}', ha='center', va='bottom')
        plt.title(title, fontsize=16)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.show()
                                        
    '''
    Accessor for bandits at the given index
    @param int key The index of the bandit you want to access
    @return Bandit The bandit at the given index
    '''
    def __getitem__(self, key):
        if key < 0 or key >= len(self): raise ValueError("index out of bounds")
        if type(key) != int: raise TypeError("Expected integer index")
        return self._bandits[key]

    '''
    Accessor for an iterator for list of bandits in this environment
    @return iter An iterator for list of bandits in this environment
    '''
    def __iter__(self):
        return iter(self._bandits)

    '''
    Accessor for number of bandits in this environment
    @return int Number of bandits in this environment
    '''
    def __len__(self):
        return len(self._bandits)

'''
An Action is a choice of a bandit by a given agent. It can return a reward and determine whether the action was optimal
'''
class Action:
    '''
    Constructor for objects of type Action
    @param Agent agent The agent which chooses this action
    @param int index The index of the bandit you want to choose
    '''
    def __init__(self, agent, index):
        self._index = index
        self._agent = agent
        if index >= len(self._agent.getEnvironment()) or index < 0:
            raise ValueError("index out of range")

    '''
    Decides if this action was the most likely to return the highest reward
    @return true if this action was the most likely to return the highest reward. false, otherwise
    '''
    def isOptimal(self):
        return all([self._agent.getEnvironment()[self._index].getMean() >= bandit.getMean() for bandit in self._agent.getEnvironment()])

    '''
    Get the reward from taking this action
    @return float The reward from taking this action
    '''
    def getReward(self):
        return self._agent.getEnvironment()[self._index].spin()

'''    
An Agent is an actor which attempts to choose the best bandit in the given environment based on the policy determined by the given value of epsilon
'''
class Agent:

    '''
    Constructor for objects of type Agent
    @param float epsilon The rate at which this agent will choose to randomly explore other options despite expected rewards
    @param Environment environment The environment in which this agent will act
    '''
    def __init__(self, epsilon, environment):
        self._epsilon = epsilon
        self._environment = environment
        
    '''
    Choose the bandit with the highest expected reward
    @return Action The action with the highest expected reward
    '''
    def exploit(self):
        return Action(self, max(range(len(self._environment)), key=lambda index: self._environment[index].estimate()))

    '''
    Randomly choose an action despite expected rewards
    @return Action A random action possible in the environment
    '''
    def explore(self):
        return Action(self, random.randint(0, len(self._environment) - 1))
    
    '''
    Choose an action based on policy determined by this agents value of epsilon
    @return Action An action based on the policy of this agent
    '''
    def createAction(self):
        if random.random() < self._epsilon:
            return self.explore()
        else:
            return self.exploit()
    
    '''
    Mutator for the value of epsilon for this agent
    @param float epsilon The new value of epsilon you want to set for this agent
    '''
    def setEpsilon(self, epsilon):
        self._epsilon = epsilon
        
    '''
    Accessor for the value of epsilon for this agent
    @return float The value of epsilon for this agent
    '''
    def getEpsilon(self):
        return self._epsilon

    '''
    Accessor for the environment that this agent exists in
    @return Environment The environment that this agent exists in
    '''
    def getEnvironment(self):
        return self._environment

    '''
    Mutator for the environment that this agent exists in
    @param Environement environment The environment you want this agent to exist in
    '''
    def setEnvironment(self, environment):
        self._environment = environment
'''
A Train is a class to manage the training of agents and plotting the results
'''
class Train:
    fig = plt.figure(figsize=(10,6))

    '''
    # Constructor for objects of type Train
    # @param int num_agents The num of agents to be trained
    # @param int iterations The number of iterations for each agent in the training process
    '''
    def __init__(self, num_agents=1000, iterations=2000, num_bandits=10):
        if type(num_agents) != int: raise TypeError("Expected integer number of agents")
        if type(iterations) != int: raise TypeError("Expected integer number of iterations")
        self._num_bandits = num_bandits
        self._num_agents = num_agents
        self._iterations = iterations
        
    '''
    Train the number of agents over the number of iterations associated with this train
    @param float/lambda epsilon The epsilon associated with every agent. May be a function of time
    @param float initial_expectation The initial estimate of the reward from every bandit. 0, by default
    @param lambda update_rule The update rule when estimating the reward associated with a bandit. 1/n (uniform average), by default
    @param lambda grad_shift The function by which the mean of each bandit in the environment of each agent is changed each iteration. 0, by default
    @param float perm_prob The probability that the collection of bandits will permute their mean value. 0, by default
    @param int perm_step The step at which the collection of bandits will permute their mean values. None, by default
    @param float new_bandits_prob The probability that the collection of bandits will be giving new mean values. 0, by default
    @param int new_bandits_step The step at which the collectiion of bandits will be given new mean values. None by default
    @param boolean return_percent Set this to True if you want the average, percent optimal choice over time. False if you want the average reward of the agents over time
    @return list, list The average reward of the agents over time, The average, percent optimal choice over time
    '''
    def train_agents(self, epsilon, initial_expectation=0, update_rule=lambda n: 0 if n==0 else 1/n, grad_shift=None, perm_prob=0, perm_step=None, new_bandits_prob=0, new_bandits_step=None, return_percent=False):
        np.random.seed(0) # Set constant seed for consistent reproduciblity of results
        
        if type(return_percent) != bool: raise TypeError("Expected boolean parameter return_percent")
        if type(epsilon) in [int, float]: # If epsilon is given as a constant, define constant function for concise code
            temp = epsilon
            epsilon = lambda x : temp
        if type(update_rule) in [int, float]: # If update_rule is given as a constant, define constant function for concise code
            temp = update_rule
            update_rule = lambda x : temp

        average_reward = self._iterations * [0] # Average rewards initialized
        percent_optimal = self._iterations * [0] # Percent optimal initialized
        
        for i in range(self._num_agents):
            environment = Environment(bandit_range_variance=1, num_bandits=self._num_bandits, initial_estimate=initial_expectation, alpha=update_rule)
            agent = Agent(epsilon(0), environment)
            for time in range(self._iterations):
                agent.setEpsilon(epsilon(time)) # Change epsilon according to the given function
                action = agent.createAction() # The agent chooses an action
                if grad_shift != None: self.grad_shift(environment, grad_shift, time) # Gradually shift the mean of each bandit
                if random.random() < perm_prob or time == perm_step: self.permute_means(environment) # Permute the mean values
                if random.random() < new_bandits_prob or time == new_bandits_step: self.new_bandits(environment) # Set new means
                average_reward[time] = (1 / (i+1)) * action.getReward() + ((i-1) / i) * average_reward[time] if i != 0 else action.getReward() # Track reward and keep average
                if return_percent: percent_optimal[time] = (1 / i) * int(action.isOptimal()) + ((i-1) / i) * percent_optimal[time] if i!= 0 else int(action.isOptimal()) # Track percentage of agents which made the optimal choice over time
        if return_percent: return percent_optimal       
        return average_reward
    
    
    '''
    Plot the results and assign a label
    @list reward The data you want to plot
    @str label The label assigned to the given data
    @str color The color of the plot
    '''
    def plot(self, reward, label, color=None):
        if color == None: plt.plot(reward, label=label)
        else: plt.plot(reward, label=label, color=color)
        
    '''
    Display the results of all plots
    @param str title The title of the plot
    @param str xaxis The label for the x axis of the plot
    @param str yaxis The label for the y axis of the plot
    @param str legend_loc The location of the legend. "upper left", by default
    '''
    def show_plot(self, title, xaxis, yaxis, legend_loc="upper left"):
        self.fig.show()
        plt.title(title)
        plt.xlabel(xaxis)
        plt.ylabel(yaxis)
        plt.legend(loc=legend_loc, fontsize=13)

    '''
    Shift the mean of each bandit in the given environment according to the given function
    @param Environment environment The environment in which the bandits are to be altered
    @param lambda shift A function of the previous mean and t, according to which the mean of each bandit in the given environment is to be altered
    '''
    def grad_shift(self, environment, shift, t=0):
        for bandit in environment:
                bandit.setMean(shift(bandit.getMean(), t))

    '''
    Permute the means assigned to the given environment
    @param Environment environment The environment in which the bandits are to be altered
    '''
    def permute_means(self, environment):
        means = [bandit.getMean() for bandit in environment]
        random.shuffle(means)
        for bandit, mean in zip(list(environment), means):
            bandit.setMean(mean)

    '''
    Change the mean values associated with the bandits in the given enironment
    @param Environment environment The environment in which the bandits are to be altered
    '''
    def new_bandits(self, environment):
        for bandit in list(environment):
            bandit.setMean(np.random.normal())






'''
Example Environment distributions
'''
def fig1():
    np.random.seed(10)
    env = Environment()
    env.plot_bandits()

'''
Example Bandit distribution
'''
def fig2():
    np.random.seed(10)
    mean=np.random.normal()
    bandit = Bandit(mean)
    bandit.plot_rewards()
    print(mean)


"""
Compare average rewards from values of epsilon: 0.1, 0.2, 0.01, 1, 0
"""
def fig3():
    train = Train()
    
    train.plot(train.train_agents(epsilon=0.1),label="epsilon=0.1")
    train.plot(train.train_agents(epsilon=0.2),label="epsilon=0.2")
    train.plot(train.train_agents(epsilon=0.01),label="epsilon=0.01")
    train.plot(train.train_agents(epsilon=1),label="epsilon=1")
    train.plot(train.train_agents(epsilon=0),label="epsilon=0")

    train.show_plot(title="Average Reward Over Time", xaxis="Time", yaxis="Reward")

'''
Declare epsilon as a function of time. Show plot as percent of optimal choice over time
'''
def fig4():
    train = Train()
    
    train.plot(train.train_agents(epsilon=0.1, return_percent=True),label="epsilon=0.1")
    train.plot(train.train_agents(epsilon=lambda t:(50/(t+1)), return_percent=True),label="epsilon=(50/t)")

    train.show_plot(title="Percent of Optimal Choice Over Time", xaxis="Time", yaxis="Percent of Agents Which Made Optimal Choice")

'''
Compare optimistic and pessimistic initial assumptions
'''
def fig5():
    train = Train()
    
    train.plot(train.train_agents(epsilon=0.01, return_percent=True), label="Pessimistic")
    train.plot(train.train_agents(epsilon=0.01, initial_expectation=50, return_percent=True), label="Optimistic")

    train.show_plot(title="Comparing Different Initial Assumptions", xaxis="Time", yaxis="Percent of Agents Which Made Optimal Choice")

'''
Display gradually shifting Bandit mean value
'''
def fig6():
    np.random.seed(0)
    bandit = Bandit(mean=np.random.normal())
    bandit.plot_bandit_mean(shift=lambda m:m+np.random.normal(scale=0.01))        
    bandit.show_plot(title='Mean Value of Bandit Rewards Over Time', xaxis='Time', yaxis='Mean Value', legend_loc="upper left")

'''
Compare values of epsilon with a gradual change to bandit mean rewards
'''
def fig7():
    np.random.seed(0)
    train = Train()

    train.plot(train.train_agents(epsilon=0.1, return_percent=True, grad_shift=lambda m,t:m+np.random.normal(scale=0.01)), label="epsilon=0.1")
    train.plot(train.train_agents(epsilon=0.2, return_percent=True, grad_shift=lambda m,t:m+np.random.normal(scale=0.01)), label="epsilon=0.2")
    train.plot(train.train_agents(epsilon=lambda t:50/(t+1), return_percent=True, grad_shift=lambda m,t:m+np.random.normal(scale=0.01)), label="epsilon=50/t")

    train.show_plot(title="Percent of Optimal Choice With Gradual Shift in Avg Rewards", xaxis="Time", yaxis="Percent of Agents Which Made Optimal Choice")

'''
Compare values of epsilon with a gradual change to bandit mean rewards
'''
def fig8():
    np.random.seed(0)
    train = Train()

    train.plot(train.train_agents(epsilon=0.1, return_percent=True, grad_shift=lambda m,t:m+np.random.normal(scale=0.01)), label="epsilon=0.1")
    train.plot(train.train_agents(epsilon=0.2, return_percent=True, grad_shift=lambda m,t:m+np.random.normal(scale=0.01)), label="epsilon=0.2")
    train.plot(train.train_agents(epsilon=lambda t:0.05 + 50/(t+1), return_percent=True, grad_shift=lambda m,t:m+np.random.normal(scale=0.01)), label="epsilon=0.05 + (50/t)")

    train.show_plot(title="Percent of Optimal Choice With Gradual Shift in Avg Rewards", xaxis="Time", yaxis="Percent of Agents Which Made Optimal Choice")
    
'''
Compare forgetful update rule with gradual change to bandit mean rewards
'''
def fig9():
    np.random.seed(0)
    train = Train()

    train.plot(train.train_agents(epsilon=lambda t:50/(t+1), return_percent=True, update_rule=0.1, grad_shift=lambda m,t:m+np.random.normal(scale=0.01)), label="epsilon=50/t, Forgetful Step Size")
    train.plot(train.train_agents(epsilon=lambda t:50/(t+1), return_percent=True, grad_shift=lambda m,t:m+np.random.normal(scale=0.01)), label="epsilon=50/t, Averaging Rule")

    train.show_plot(title="Percent of Optimal Choice With Gradual Shift and Forgetful Update Rule", xaxis="Time", yaxis="Percent of Agents Which Made Optimal Choice")

'''
Display mean reverting gradually shifting bandit mean values
'''
def fig10():
    np.random.seed(0)
    bandit = Bandit(mean=np.random.normal())
    bandit.plot_bandit_mean(shift=lambda m:(0.5 * m)+np.random.normal(scale=0.01), label="kappa=0.5")
    np.random.seed(0)
    bandit.setMean(np.random.normal())
    bandit.plot_bandit_mean(shift=lambda m:(0.99 * m)+np.random.normal(scale=0.01), color='b', label="kappa=0.99")
    bandit.show_plot(title='Mean Value of Bandit Rewards Over Time', xaxis='Time', yaxis='Mean Value', legend_loc="upper left")

'''
Compare values of epsilon with abrupt change to bandit mean rewards
'''
def fig11():
    np.random.seed(0)
    train = Train()
    kappa = 0.99

    train.plot(train.train_agents(epsilon=lambda t:50/(t+1), return_percent=True, update_rule=0.1, grad_shift=lambda m,t:(kappa * m)+np.random.normal(scale=0.01)), label="epsilon=50/t, Forgetful Step Size")
    train.plot(train.train_agents(epsilon=0.1, return_percent=True, update_rule=0.1, grad_shift=lambda m,t: (kappa * m)+np.random.normal(scale=0.01)), label="epsilon=0.1, Forgetful Step Size")
    train.plot(train.train_agents(epsilon=0.2, return_percent=True, update_rule=0.1, grad_shift=lambda m,t: (kappa * m)+np.random.normal(scale=0.01)), label="epsilon=0.2, Forgetful Step Size")


    train.show_plot(title="Mean Reverting Change with Forgetful Update Rule", xaxis="Time", yaxis="Percent of Agents Which Made Optimal Choice")

'''
Show permuted environment
'''
def fig12():
    np.random.seed(10)
    env = Environment()
    env.plot_bandits(permute=True, title="Sampled Reward Distribution for Each Bandit After Permutation")
    
'''
Try to optimize values of epsilon with abrupt change to bandit mean rewards
'''
def fig13():
    train = Train()

    train.plot(train.train_agents(epsilon=lambda t:(50/(t+1)), return_percent=True, perm_step=1000, update_rule=0.1), label="epsilon=(50/t)")
    train.plot(train.train_agents(epsilon=0.1, return_percent=True, perm_step=1000, update_rule=0.1), label="epsilon=0.1")

    train.show_plot(title="Percent of Optimal Choice With Abrupt Change and Forgetful Update Rule", xaxis="Time", yaxis="Percent of Agents Which Made Optimal Choice")

'''
Specially tuned epsilon assignment
'''
def fig14():
    train = Train()

    train.plot(train.train_agents(epsilon=lambda t: 1 if 1000 <= t <= 1050 else 0.05 + 50/(t+1), return_percent=True, perm_step=1000, update_rule=0.1), label="epsilon=e_t")

    train.show_plot(title="Percent of Optimal Choice With Abrupt Change and Forgetful Update Rule", xaxis="Time", yaxis="Percent of Agents Which Made Optimal Choice")













# Main script to be run
# Uncomment methods you want to run and comment those which you do not
# Displays figure according to juxtaposed number
def main():
    fig1()
##    fig2()
##    fig3()
##    fig4()
##    fig5()
##    fig6()
##    fig7()
##    fig8()
##    fig9()
##    fig10()
##    fig11()
##    fig12()
##    fig13()
##    fig14()



    
# Run program
if __name__ == "__main__":
    main()
    
    

    
