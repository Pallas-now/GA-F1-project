#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random
import numpy as np
from random import random as rnd

tyre_list = ['S', 'M', 'H']

#Australia
laps_in_total= 58
tb =84
overtake_delta = 0.8


def individual (number_of_pitstop):
    pitstop_list = random.sample(range(1,laps_in_total -1), number_of_pitstop) #random choose pitstop,l is the lap number of stop
    pitstop_list.sort()
    choices_tyre_list = [random.choice(tyre_list) for x in range(number_of_pitstop + 1)]# choose kinds of tyre
    individual = []
    for i in range(len(pitstop_list)):
        individual = individual + [choices_tyre_list[i]] + [pitstop_list[i]]
    individual.append(choices_tyre_list[-1])
    return individual

def population (number_of_individuals, number_of_pitstop):
    population = [individual(number_of_pitstop) for x in range(number_of_individuals)]
    return population

#tl is the lap time, tb is the base time, corresponding to the perfect/fastest possible lap, pt is the time 
#penalty due to tyre wear and pf is the time penalty due to the fuel load onboard

def fitness_calculation (individual, number_of_pitstop):
    #divide individual into pitstop_list and choices_tyre_list
    pitstop_list = [individual[i] for i in range(1, len(individual), 2)]
    pitstop_list.sort()
    choices_tyre_list = [individual[i] for i in range(0, len(individual), 2)]
    #list of lap length
    stint_list = [pitstop_list[0]] + [pitstop_list[i+1] - pitstop_list[i] for i in range(len(pitstop_list)-1)] + [laps_in_total - pitstop_list[-1]]
    #print(laps_list)

    #time penalty due to tyre wear
    total_race_time = 0
    for x in range(len(stint_list)):
        lap_time = 0
        #stint time corresponding to a stint of laps_list[x] laps
        for l in range(stint_list[x]):#laps_list[x] is the lMax of each stint
            if choices_tyre_list[x] == 'S':
                pt = 0.004 * l **2
            elif choices_tyre_list[x] == 'M':
                pt = 0.02 * l + 0.2
            elif choices_tyre_list[x] == 'H': 
                pt = 0.4
        #time penalty due to the fuel load onboard
            pf = 0.08 * (stint_list[x] - l)
            lap_time += tb + pt + pf
        total_race_time += lap_time 
    
    tp = 0
    for x in range(1,len(stint_list)):
        tp += 16 + 0.5 * stint_list[x] #a pit-stop costs tp
        
    total_race_time = total_race_time + tp
    return total_race_time

individual(2)


# In[2]:


# Creating the First Generation
def first_generation(pop, number_of_pitstop):
    fitness = [fitness_calculation(pop[x], number_of_pitstop) 
        for x in range(len(pop))]
    sorted_fitness = sorted([[pop[x], fitness[x]]
        for x in range(len(pop))], key=lambda x: x[1]) # sorted by fitness[x] with ascending order
    population = [sorted_fitness[x][0] 
        for x in range(len(sorted_fitness))]
    fitness = [sorted_fitness[x][1] 
        for x in range(len(sorted_fitness))]
    return {'Individuals': population, 
            'Fitness': fitness}

p1 = first_generation(population(10,2), 2)
p1


# In[3]:


#selesction, method is fittest half selection

def selection (generation, method='Fittest Half'):
    if method == 'Fittest Half':
        selected_individuals = [generation['Individuals'][x]
            for x in range(int(len(generation['Individuals'])//2))]
        selected_fitnesses = [generation['Fitness'][x]
            for x in range(int(len(generation['Individuals'])//2))]
        selected = {'Individuals': selected_individuals,
                    'Fitness': selected_fitnesses}
    elif method == 'Random':
        random_number = random.sample(range(len(generation['Fitness'])),int(len(generation['Individuals'])//2))
        selected_individuals =             [generation['Individuals'][x]
            for x in random_number]
        selected_fitnesses = [generation['Fitness'][x]
            for x in random_number]
        selected = {'Individuals': selected_individuals,
                    'Fitness': selected_fitnesses}
    return selected

selected = selection (p1)
selected


# In[4]:


def pairing(elit, selected, method = 'Fittest'):
    individuals = [elit['Individuals']] + selected['Individuals']
    fitness = [elit['Fitness']] + selected['Fitness']
    if method == 'Fittest':
        pair_parents = [[individuals[x],individuals[x+1]] 
                   for x in range(len(individuals)//2)]
        pair_parents.append([individuals[0],individuals[len(individuals)//2]])
        
    if method == 'Random':
        pair_parents = []
        for x in range(-(-len(individuals)//2)):
            pair_parents.append(
                [individuals[random.randint(0,(len(individuals)-1))],
                 individuals[random.randint(0,(len(individuals)-1))]])
            while pair_parents[x][0] == pair_parents[x][1]:
                pair_parents[x][1] = individuals[
                    random.randint(0,(len(individuals)-1))]
    return pair_parents


# In[5]:


def mating(parents, method='Single Point'):
    if method == 'Single Point':
        pivot_point = random.randint(1, len(parents[0]))
        offsprings = [parents[0]             [0:pivot_point]+parents[1][pivot_point:]]
        offsprings.append(parents[1]
            [0:pivot_point]+parents[0][pivot_point:])
    if method == 'Two Pionts':
        pivot_point_1 = random.randint(1, len(parents[0])-1)
        pivot_point_2 = random.randint(1, len(parents[0]))
        while pivot_point_2 < pivot_point_1:
            pivot_point_2 = random.randint(1, len(parents[0]))
        offsprings = [parents[0][0:pivot_point_1]+
            parents[1][pivot_point_1:pivot_point_2]+
            parents[0][pivot_point_2:]]
        offsprings.append(parents[1][0:pivot_point_1]+
            parents[0][pivot_point_1:pivot_point_2]+
            parents[1][pivot_point_2:])
    return offsprings


# In[6]:


def mutation(individual, limit, mutation_cutoff = 0.5):#The mutation of pitstop is in range(pitstop- limit, pitstop + limit)
    random_number = []
    mutated_individual = individual.copy()
    for i in range(len(individual)):
        #random number for each gene
        random_number.append(random.uniform(0,1))
    for i in range(0, len(individual), 2):
        if random_number[i] < mutation_cutoff:
            mutated_individual[i] = random.choice(tyre_list)
    for i in range(1, len(individual), 2):
        if random_number[i] < mutation_cutoff:
            mutated_individual[i] = random.randint(individual[i] - limit, individual[i] + limit)
            while mutated_individual[i] > laps_in_total:
                mutated_individual[i] = random.randint(individual[i] - limit, laps_in_total-1)     
    return mutated_individual


# In[7]:


parents = [[['M', 24, 'M', 40, 'S'], ['M', 7, 'H', 35, 'H']],
 [['M', 7, 'H', 35, 'H'], ['M', 27, 'H', 48, 'S']]]


offsprings = [mating(parents[x],'Two Pionts')
                    for x in range(len(parents))]
offsprings


# In[8]:


def next_generation(gen, limit, number_of_pitstop, mutation_cutoff = 0.5):#The mutation of pitstop is in range(pitstop- limit, pitstop + limit)
    elit = {}
    next_gen = {}
    #elitism best individual in a generation moves on to the next generation without mutating
    elit['Individuals'] = gen['Individuals'].pop(0)
    elit['Fitness'] = gen['Fitness'].pop(0)
    selected = selection(gen)
    pairs_parents = pairing(elit, selected)
    offsprings = [mating(pairs_parents[x]) 
                  for x in range(len(pairs_parents))]
    offsprings1 = [offsprings[x][0]
                   for x in range(len(pairs_parents))]
    offsprings2 = [offsprings[x][1]
                   for x in range(len(pairs_parents))]
    unmutated = selected['Individuals'] + offsprings1 + offsprings2
    mutated = [mutation(unmutated[x], limit, mutation_cutoff) 
        for x in range(len(unmutated))]
    unsorted_individuals = mutated + [elit['Individuals']]
    unsorted_fitness =         [fitness_calculation(mutated[x], number_of_pitstop) 
         for x in range(len(mutated))] + [elit['Fitness']]
    sorted_next_gen =         sorted([[unsorted_individuals[x], unsorted_fitness[x]]
            for x in range(len(unsorted_individuals))], 
                key=lambda x: x[1])
    next_gen['Individuals'] = [sorted_next_gen[x][0]
        for x in range(len(sorted_next_gen))]
    next_gen['Fitness'] = [sorted_next_gen[x][1]
        for x in range(len(sorted_next_gen))]
    #dedupicate
    list(next_gen)
    gen['Individuals'].append(elit['Individuals'])
    gen['Fitness'].append(elit['Fitness'])
    return next_gen
  


# In[9]:


# limit the number for the same individual to be the best individual
def fitness_similarity_chech(min_fitness, number_of_similarity):#min_fitness is a np array
    result = False
    similarity = 0
    for n in range(len(min_fitness)-1):
        if min_fitness[n] == min_fitness[n+1]:
            similarity += 1
        else:
            similarity = 0
    if similarity == number_of_similarity-1:
        result = True
    return result


# In[25]:


import numpy as np
# Generations and fitness values will be written to this file
Result_file = 'G:/R/dissertation/GA_Results.txt'
Best_result_file =  'G:/R/dissertation/GA_Best_Results.txt'# Best result of each run

number_of_individuals = 10
number_of_pitstop = 2
number_of_similarity = 25
limit = 10
mutation_cutoff = 0.5

end = False
i = 1
number_of_run = 100
number_of_gen = []# list of Number of gengeration of each run
best_individual = []# list of best_individual of each run
best_fitness = []# list of best_fitness of each run

while end == False:
    if i > number_of_run:
        break
    pop = population(number_of_individuals, number_of_pitstop)#number of individul, number_of_pitstop
    gen = []
    gen.append(first_generation(pop, number_of_pitstop)) #pop, number_of_pitstop
    fitness_min = np.array([min(gen[0]['Fitness'])])
    
    res = open(Result_file, 'a')
    res.write('\n'+ 'Number of run: ' + str(i) + '\n' +'\n'+              str(gen) +'\n')
    res.close()
    
    finish = False
    number_of_gengeration = 0
    
    while finish == False:
        if fitness_similarity_chech(fitness_min, number_of_similarity) == True:
            break
        gen.append(next_generation(gen[-1], limit, number_of_pitstop)) # limit = 10, pitstop = 1, mutation cutoff: 0.5
        fitness_min = np.append(fitness_min, min(gen[-1]['Fitness']))
        res = open(Result_file, 'a')
        res.write('\n' + str(gen[-1]) + '\n')
        number_of_gengeration += 1
        res.close()
        
    res = open(Result_file, 'a')
    res.write('\n'+ '---End---'+ '\n'+'\n')
    res.close()
    
    res = open(Best_result_file, 'a')
    res.write('\n'+ 'Number of run: ' + str(i) + '\n'+               'Number of pitstop: ' + str(number_of_pitstop) + '\n' +           'Range of mutation: ' + str(limit) + '\n' +           'Mutation cutoff: ' + str(mutation_cutoff) +'\n' +           'Fitness similarity: ' + str(number_of_similarity) + '\n'+           'Number of gengeration: ' + str(number_of_gengeration) +'\n' +           'The best individual: ' +  str(gen[-1]['Individuals'][0]) + '\n' +           'The time: ' + str(gen[-1]['Fitness'][0]) + '\n' +          '---End---'+ '\n'+'\n')
    res.close()
    
    number_of_gen.append(number_of_gengeration)
    best_individual.append(gen[-1]['Individuals'][0])
    best_fitness.append(gen[-1]['Fitness'][0])
    i += 1

#for the total runs 
sorted_fitness_run = sorted([[best_individual[x], best_fitness[x]]
        for x in range(len(pop))], key=lambda x: x[1]) # sorted by fitness[x] with ascending order 
best_individual_run = [sorted_fitness_run[x][0] 
        for x in range(len(sorted_fitness_run))]
best_fitness_run = [sorted_fitness_run[x][1] 
        for x in range(len(sorted_fitness_run))]
best = {'Individuals': best_individual_run, 
            'Fitness': best_fitness_run}

res = open(Best_result_file, 'a')
res.write('\n'+ 'The average number of generation: ' + str(np.mean(number_of_gen)) + '\n' +          'The best individual: ' +  str(best['Individuals'][0]) + '\n' +           'The time: ' + str(best['Fitness'][0]) + '\n' +           '---End---'+ '\n'+'\n')
res.close()


    


# In[11]:


from scipy import stats
#null hypothesis that the data was drawn from a normal distribution
#Returns statistic The test statistic. p-value
stats.shapiro(number_of_gen)

