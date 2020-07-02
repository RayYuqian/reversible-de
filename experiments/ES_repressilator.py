import os
import sys
import time
import pickle
import random
import matplotlib
import math

from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

PYTHONPATH = '/Users/jmt/Dev/github/life/experiments'

sys.path.append(os.path.dirname(os.path.expanduser(PYTHONPATH)))

from algorithms.population_algorithm import OptimizationAlgorithm, EvolutionStrategy

from testbeds.repressilator_testbed import Repressilator

if __name__ == '__main__':

    # INIT: general hyperparams
    D = 4

    pop_size = 20

    num_generations = 20

    num_repetitions = 10

    # es_types = ['es_a1', 'es_a2', 'es_a3', 'es_a4', 'es_a5', 'es_a6', 'es_a7', 'es_a8'
    #            'es_b1', 'es_b2', 'es_b3', 'es_b4', 'es_b5', 'es_b6', 'es_b7', 'es_b8']
    # 'es_a7', 'es_a8', 'es_b7', 'es_b8'

    es_types = ['es_a1', 'es_a2', 'es_a3', 'es_a4', 'es_a5', 'es_a6', 'es_a7', 'es_a8']

    results_dir = '../results/Repressilator' + '_pop_' + str(pop_size) + '_allA_s1.5_v2'

    final_results = {}

    for es_type in es_types:
        print(f"------- Now runs: {es_type} -------")
        for rep in range(num_repetitions):
            print(f"\t-> repetition {rep}")

            np.random.seed(seed=rep)

            r = Repressilator()
            y_real, params = r.create_data(pop_size=pop_size)
            objective = r.objective

            params['evaluate_objective_type'] = 'single'
            params['pop_size'] = pop_size
            params['SIGMA'] = 1.5
            params['BETA'] = 5 / 180 * math.pi
            params['ALPHA'] = math.pi
            params['update'] = 0.85

            es = EvolutionStrategy(objective,
                                   args=(params, y_real),
                                   x0=params['x0'],
                                   bounds=(params['bounds'][0], params['bounds'][1]),
                                   population_size=pop_size,
                                   es_type=es_type)

            opt_alg = OptimizationAlgorithm(pop_algorithm=es, num_epochs=num_generations)

            specific_folder = '-' + es_type + '-pop_size-' + str(
                params['pop_size']) + '-epochs-' + str(num_generations)
            directory_name = results_dir + '/' + opt_alg.name + specific_folder + '-r' + str(rep)

            if os.path.exists(directory_name):
                directory_name = directory_name + str(datetime.now())

            directory_name = directory_name + '/'
            os.makedirs(directory_name)

            tic = time.time()
            res, f = opt_alg.optimize(directory_name=directory_name)
            toc = time.time()

            # Plot of best results
            f_best = np.load(directory_name + 'f_best.npy')

            plt.plot(np.arange(0, len(f_best)), np.array(f_best))
            plt.grid()
            plt.savefig(directory_name + '/' + 'best_f.pdf')
            plt.close()

            print('\tTime elapsed=', toc - tic)

        # Average best results
        directory_name_avg = directory_name[:-4]
        for r in range(num_repetitions):
            dir = directory_name_avg + '-r' + str(r)
            if r == 0:
                f_best_avg = np.load(dir + '/' + 'f_best.npy')
            else:
                f_best_avg = np.concatenate((f_best_avg, np.load(dir + '/' + 'f_best.npy')), 0)

        f_best_avg = np.reshape(f_best_avg, (num_repetitions, num_generations + 1))

        # plotting
        x_epochs = np.arange(0, f_best_avg.shape[1])
        y_f = f_best_avg.mean(0)
        y_f_std = f_best_avg.std(0)

        final_results[es_type + '_avg'] = y_f
        final_results[es_type + '_std'] = y_f_std

        plt.plot(x_epochs, y_f)
        plt.fill_between(x_epochs, y_f - y_f_std, y_f + y_f_std)
        plt.grid()
        plt.savefig(results_dir + '/' + opt_alg.name + es_type + '_best_f_avg.pdf')
        plt.close()

    # save final results (just in case!)
    f = open(results_dir + '/' + 'repressilator.pkl', "wb")
    pickle.dump(final_results, f)
    f.close()

    colors = ['#ffe119', '#3cb44b', '#4363d8', '#fc3503', '#8442f5', '#f59642']
    linestyles = ['-', '-.', 'dotted', 'dotted', '-', '-.']
    # labels = ['ES_a1', 'ES_a2', 'ES_a3', 'ES_a4', 'ES_a5', 'ES_a6']
    # ['ES_b1', 'ES_b2', 'ES_b3', 'ES_b4', 'ES_b5', 'ES_b6']
    # 'ES_a7', 'ES_a8', 'ES_b7', 'ES_b8'
    labels = ['ES_a1', 'ES_a2', 'ES_a3', 'ES_a4', 'ES_a5', 'ES_a6']
    lw = 3.
    iter = 0

    for es_type in es_types:
        plt.plot(x_epochs, final_results[es_type + '_avg'], colors[iter], ls=linestyles[iter], lw=lw,
                 label=labels[iter])
        plt.fill_between(x_epochs, final_results[es_type + '_avg'] - final_results[es_type + '_std'],
                         final_results[es_type + '_avg'] + final_results[es_type + '_std'],
                         color=colors[iter], alpha=0.5)

        iter += 1

    plt.grid()
    plt.xlabel('epochs')
    plt.ylabel('objective')
    plt.legend(loc=0)
    plt.savefig(results_dir + '/' + '_best_f_comparison_A.pdf')
    plt.close()

    # ------------------------------------------------------------------------------


