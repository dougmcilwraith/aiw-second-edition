#7.1
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import itertools

#Assume 5000 samples per group
n_experiment = 10000
n_control = 10000

p_experiment= 0.002
p_control = 0.001

se_experiment_sq = p_experiment*(1-p_experiment) / n_experiment
se_control_sq = p_control*(1-p_control) / n_control

Z = (p_experiment-p_control)/math.sqrt(se_experiment_sq+se_control_sq)

print Z

#Not a listing, but mentioned in the text and is required for production.
def get_z(n_experiment, n_control, p_experiment, p_control):
    var_experiment = p_experiment*(1-p_experiment) / n_experiment
    var_control = p_control*(1-p_control) / n_control
    Z = (p_experiment-p_control)/math.sqrt(var_experiment+var_control)
    return Z

experiment_array = np.linspace(100, 20000, 100)
control_array = np.linspace(100,20000,100)
experiment_probability_array = np.empty(100); experiment_probability_array.fill(0.002)
control_probability_array = np.empty(100); control_probability_array.fill(0.001)
data =  zip(experiment_array,control_array,experiment_probability_array,control_probability_array)
#Need to create associated parameters and zip these together.
z_results = [get_z(k[0],k[1],k[2],k[3]) for k in data]

plt.plot(experiment_array,z_results)
#95 % confidence interval
x_values = [0,20000]
y_values = [1.96,1.96]
plt.plot(x_values,y_values)
plt.text(x_values[0],y_values[0]+0.01,"95%")

#90% confidence interval
x_values = [0,20000]
y_values = [1.645,1.645]
plt.plot(x_values,y_values)
plt.text(x_values[0],y_values[0]+0.01,"90%")

#80% confidence interval 1.28
x_values = [0,20000]
y_values = [1.28,1.28]
plt.plot(x_values,y_values)
plt.text(x_values[0],y_values[0]+0.01,"80%")

#70% confidence interval 1.04
x_values = [0,20000]
y_values = [1.04,1.04]
plt.plot(x_values,y_values)
plt.text(x_values[0],y_values[0]+0.01,"70%")

plt.xlabel("Number of users in each A/B group")
plt.ylabel("z value")

plt.title("Graph of number of users against z value for a fixed conversion rate \n (0.001/0.002 A/B respectively)")
plt.show()

##Bayesian Bandit
#7.6
from scipy.stats import beta
import numpy as np
import matplotlib.pyplot as plt
import random
from operator import sub, div, add

class Bandit:
    def __init__(self,probability):
        self.probability=probability
    
    def pull_handle(self):
        if random.random()<self.probability:
            return 1
        else:
            return 0
        
    def get_prob(self):
        return self.probability

#7.7
def sample_distributions_and_choose(bandit_params):
    sample_array =  \
	[beta.rvs(param[0], param[1], size=1)[0] for param in bandit_params]
    return np.argmax(sample_array)

#7.8
def run_single_regret(bandit_list,bandit_params,plays):
    sum_probs_chosen=0
    opt=np.zeros(plays)
    chosen=np.zeros(plays)
    bandit_probs = [x.get_prob() for x in bandit_list]
    opt_solution = max(bandit_probs)
    for i in range(0,plays):
        index = sample_distributions_and_choose(bandit_params)
        sum_probs_chosen+=bandit_probs[index]
        if(bandit_list[index].pull_handle()):
            bandit_params[index]=\
		(bandit_params[index][0]+1,bandit_params[index][1])
        else:
            bandit_params[index]=\
		(bandit_params[index][0],bandit_params[index][1]+1)
        opt[i] = (i+1)*opt_solution
        chosen[i] = sum_probs_chosen
    regret_total = map(sub,opt,chosen)
    return regret_total

#7.9
#Plot params beforehand
bandit_list = [Bandit(0.1),Bandit(0.3),Bandit(0.8)]
bandit_params = [(1,1),(1,1),(1,1)]

x = np.linspace(0,1, 100)
plt.plot(x, beta.pdf(x, bandit_params[0][0], bandit_params[0][1]),'-r*', alpha=0.6, label='Bandit 1')
plt.plot(x, beta.pdf(x, bandit_params[1][0], bandit_params[1][1]),'-b+', alpha=0.6, label='Bandit 2')
plt.plot(x, beta.pdf(x, bandit_params[2][0], bandit_params[2][1]),'-go', alpha=0.6, label='Bandit 3')
plt.legend()
plt.xlabel("payout probability")
plt.ylabel("probability density of belief")
plt.show()

#7.10
#Just do a single regret
plays=1000
bandit_list = [Bandit(0.1),Bandit(0.3),Bandit(0.8)]
bandit_params = [(1,1),(1,1),(1,1)]

regret_total = run_single_regret(bandit_list,bandit_params,plays)
plt.plot(regret_total)
plt.title("expected regret against steps in experiment")
plt.xlabel("Step in experiment")
plt.ylabel("Cumulative expected regret in experiment at this step")
plt.show()

#7.11
#Now plot the params:
x = np.linspace(0,1, 100)
plt.plot(x, beta.pdf(x, bandit_params[0][0], bandit_params[0][1]),'-r*', alpha=0.6, label='Bandit 1')
plt.plot(x, beta.pdf(x, bandit_params[1][0], bandit_params[1][1]),'-b+', alpha=0.6, label='Bandit 2')
plt.plot(x, beta.pdf(x, bandit_params[2][0], bandit_params[2][1]),'-go', alpha=0.6, label='Bandit 3')
plt.legend()
plt.xlabel("payout probability")
plt.ylabel("probability density of belief")
plt.show()

#7.12
#Do many regrets on the same graph
plays=1000
runs=100

for i in range(0,runs):
    bandit_list = [Bandit(0.1),Bandit(0.3),Bandit(0.8)]
    bandit_params = [(1,1),(1,1),(1,1)]
    regret_total = run_single_regret(bandit_list,bandit_params,plays)
    plt.plot(regret_total,label='%s'%i)

plt.title("expected regret against steps in experiment")
plt.xlabel("Step in experiment")
plt.ylabel("Cumulative expected regret in experiment at this step")
plt.show()

#7.13
#Plot the average regret at each step over many runs.
regret_sum=np.zeros(plays)

plays=1000
runs=100

for i in range(0,runs):
    bandit_list = [Bandit(0.1),Bandit(0.3),Bandit(0.8)]
    bandit_params = [(1,1),(1,1),(1,1)]
    regret_total = run_single_regret(bandit_list,bandit_params,plays)
    regret_sum=map(add,regret_sum, np.asarray(regret_total))

plt.plot(regret_sum/(runs*np.ones(plays)))
plt.title("average expected regret at each step over %s iterations"%runs )
plt.xlabel("Step in experiment")
plt.ylabel("Average total expected regret in experiment at this step")
plt.show()

#From here on, none of this code is reproduced within the book, however it is used to derive figures in the book.
def plot_regret(p_a,p_b,p_c,marker_cycle):
    plays=1000
    runs=100
    regret_sum=np.zeros(plays)
    for i in range(0,runs):
        bandit_list = [Bandit(p_a),Bandit(p_b),Bandit(p_c)]
        bandit_params = [(1,1),(1,1),(1,1)]
        regret_total = run_single_regret(bandit_list,bandit_params,plays)
        regret_sum = map(add,regret_sum, np.asarray(regret_total))
    plt.plot(regret_sum/(runs*np.ones(plays)),label="%s,%s,%s"%(p_a,p_b,p_c),marker=marker_cycle.next(),markevery=50)
    plt.title("average expected regret at each step over %s iterations"%runs)

def plot_regret_number_bandits(number_bandits,color_cycle,marker_cycle):
    plays=1000
    runs=100
    regret_sum=np.zeros(plays)
    for i in range(0,runs):
        bandit_list = []
        bandit_params = []
        for j in np.linspace(1,0,number_bandits, endpoint=False): #dont want to include 0 prob bandit
            bandit_list.append(Bandit(j))
            bandit_params.append((1,1))
        regret_total = run_single_regret(bandit_list,bandit_params,plays)
        regret_sum = map(add,regret_sum,np.asarray(regret_total))
    plt.plot(regret_sum/(runs*np.ones(plays)),label="# Bandits: %s"%(number_bandits),color=color_cycle.next(),marker=marker_cycle.next(),markevery=50)

#Experiment for regret curve where probabiltiies are closer to each other
markers = itertools.cycle((',', '+', '.', 'o', '*')) 

plot_regret(0,0.5,1,markers)
plot_regret(0.1,.5,.9,markers)
plot_regret(.2,.5,.8,markers)
plot_regret(.3,.5,.7,markers)
plot_regret(.4,.5,.6,markers)

plt.legend()
plt.xlabel("Step in experiment")
plt.ylabel("Average total expected regret in experiment at this step")
plt.show()

#Experiment for regret curve where probabilities are on a different scale

plot_regret(0,0.05,0.1,markers)
plot_regret(0.01,0.05,0.09,markers)
plot_regret(0.02,0.05,0.08,markers)
plot_regret(0.03,0.05,0.07,markers)
plot_regret(0.04,0.05,0.06,markers)

plt.legend()
plt.xlabel("Step in experiment")
plt.ylabel("Average cumulated expected regret in experiment at this step")
plt.show()

markers = itertools.cycle((',', '+', '.', 'o', '*', '^', 'D', 'h', 'p', 's')) 

#Experiment for regret curve where the number of bandits are increased.
color_cycle = itertools.cycle(plt.cm.spectral(np.linspace(0,1,10)))
plot_regret_number_bandits(1,color_cycle,markers)
plot_regret_number_bandits(2,color_cycle,markers)
plot_regret_number_bandits(3,color_cycle,markers)
plot_regret_number_bandits(4,color_cycle,markers)
plot_regret_number_bandits(5,color_cycle,markers)
plot_regret_number_bandits(6,color_cycle,markers)
plot_regret_number_bandits(7,color_cycle,markers)
plot_regret_number_bandits(8,color_cycle,markers)
plot_regret_number_bandits(9,color_cycle,markers)
plot_regret_number_bandits(10,color_cycle,markers)

plt.legend()
plt.xlabel("Step in experiment")
plt.ylabel("Average cumulative expected regret experiment at this step")
plt.show()
print "end"
