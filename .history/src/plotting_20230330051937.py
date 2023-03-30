import numpy as np
import pandas as pd
import os
import sys

import matplotlib.pyplot as plt


def plot_last_iter( df1, x_name,  y_name, x_axis, y_axis,index_name,  save_path):
    # plot y of last iteration against x
    plt.clf()
    index_list = pd.unique(df1[index_name])
    print(index_list)
    for index in index_list:
        if index >= 0:
            df2 = df1.loc[df1[index_name] == index]
            #print(df2)

            mean = np.array(df2[y_name+'_mean'])
            se = np.array(df2[y_name+'_se'])

            x = np.array(df2[x_name])
            
            print(f"index: {index},  x: {x}, y: {mean}, se: {se}")
            
            ################variance
            if x_name == 'prior_sd':
                x = x**2
            ################

            scale = 2.0
            lower = mean - scale * se
            upper = mean + scale * se

            

            plt.fill_between(x, lower, upper, alpha=0.2)
            plt.plot(x, mean, label=index_name+'='+str(index))

    
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.legend(loc = 'best', prop={'size': 6})
    plt.savefig(save_path, dpi = 600)




def plot_sample_path( df, x_name,  y_name, x_axis, y_axis,index_name_1, index_name_2,  save_path , gamma = None):
    # plot y along the sample path, for each index_1 index_2 pair
    # one plot per each index_1, multiple index_2 in each plot

    plt.clf()

    index_list_1 = pd.unique(df[index_name_1]) 

    for index_1 in index_list_1:
        if index_1 >= 0:
            df1 = df.loc[df[index_name_1] == index_1]

            index_list_2 = pd.unique(df1[index_name_2])
            plt.clf()

            for index_2 in index_list_2:
                if index_2 >= 0:


                    df2 = df1.loc[df1[index_name_2] == index_2]
                    horizon = df2.shape[0]

                    mean = df2[y_name+'_mean']
                    se = df2[y_name+'_se']

                    if y_name == 'TS-Bayes_erors' or y_name == 'greedy_errors':
                        if gamma is not None:
                            x = horizon/(np.arange(horizon)+1)*gamma
                            plt.xlim([gamma/2, 20])
                            plt.xscale('log')
                            x_axis = '$\gamma$'
                            y_axis = r'$\|\|\hat{\beta} - \beta\|\|_2^2$'
                        elif index_name_1 == 'gamma':
                            x = horizon/(np.arange(horizon)+1)*index_1
                            plt.xlim([gamma/2, 20])
                            plt.xscale('log')
                            x_axis = '$\gamma$'
                            y_axis = r'$\|\|\hat{\beta} - \beta\|\|_2^2$'
                        else:
                            x = np.arange(horizon)
                            y_axis = r'$\|\|\hat{\beta} - \beta\|\|_2^2$'
                    else:
                        x = np.arange(horizon)


                    scale = 2.0
                    lower = mean - scale * se
                    upper = mean + scale * se

                    plt.fill_between(x, lower, upper, alpha=0.2)
                    plt.plot(x, mean, label=index_name_2+'='+str(index_2))
                    if y_name == 'TS-Bayes_erors' and np.max(mean)>10:
                        plt.ylim(0, 10)
                    if y_name == 'greedy_errors' and np.max(mean)>10:
                        plt.ylim(0, 10)


            plt.xlabel(x_axis)
            plt.ylabel(y_axis)
            plt.legend(loc = 'best', prop={'size': 6})
            plt.savefig(f"{save_path}-{index_name_1}-{str(index_1)}-2.jpg", dpi = 600)


def plot_statistics(output_folder_name, output_name, figure_folder_name, mode,  gamma = None):
    #method_list = ['TS-Bayes_errors', 'TS-Bayes_regret','TS-Bayes_cumregret','TS-Bayes_lambda_mins', 'TS-Bayes_lambda_maxs','TS-Bayes_projs_first','TS-Bayes_log_maxs_over_mins', 'TS-Bayes_thinnesses',  'TS-Bayes_lambdas_second', 'TS-Bayes_lambdas_third', 'TS-Bayes_lambdas_d_minus_1', 'TS-Bayes_lambdas_half_d', 'TS-Bayes_biases', 'TS-Bayes_variances','TS-Bayes_risks']+['greedy_errors','greedy_regret','greedy_cumregret','greedy_lambda_mins', 'greedy_lambda_maxs','greedy_projs_first','greedy_log_maxs_over_mins', 'greedy_thinnesses',  'greedy_lambdas_second', 'greedy_lambdas_third', 'greedy_lambdas_d_minus_1', 'greedy_lambdas_half_d', 'greedy_biases', 'greedy_variances','greedy_risks']
    #method_list = ['TS-Bayes_errors', 'TS-Bayes_errors_candidate', 'TS-Bayes_errors_pcandidate', 'TS-Bayes_regret','TS-Bayes_cumregret','TS-Bayes_lambda_mins', 'TS-Bayes_lambda_maxs','TS-Bayes_projs_first', 'TS-Bayes_thinnesses',  ]\
    #+['greedy_errors', 'greedy_errors_candidate', 'greedy_errors_pcandidate','greedy_regret','greedy_cumregret','greedy_lambda_mins', 'greedy_lambda_maxs','greedy_projs_first', 'greedy_thinnesses',]\
    #+     ['TS-ThinnessDirected_errors','TS-ThinnessDirected_regret','TS-ThinnessDirected_cumregret','TS-ThinnessDirected_lambda_mins', 'TS-ThinnessDirected_lambda_maxs','TS-ThinnessDirected_projs_first', 'TS-ThinnessDirected_thinnesses',]\
    #+     ['TS-Thinness_errors','TS-Thinness_regret','TS-Thinness_cumregret','TS-Thinness_lambda_mins', 'TS-Thinness_lambda_maxs','TS-Thinness_projs_first', 'TS-Thinness_thinnesses',]\
    attributes  = ['errors', 'regret','cumregret','approx_alphas','worst_alphas' ,'lambda_mins', 'lambda_maxs','lambda_maxs_over_mins','projs_first', 'thinnesses',  'betas', 'zetas']
    method_list  = ['TS_'+atti for atti in attributes] 


    for method in method_list:
        save_path = f'{figure_folder_name}/{method}/'
        isExist = os.path.exists(save_path)
        if not isExist:
            os.makedirs(save_path)

        df = pd.read_csv(f"{output_folder_name}/all-last-{output_name}.csv")
        plot_last_iter(df,mode ,method,mode,'y','k',f"{save_path}{method}-{output_name}'-1.jpg" )

        df = pd.read_csv(f"{output_folder_name}/all-{output_name}.csv")
        plot_sample_path(df, mode, method, 't', 'y', mode, 'k', f'{save_path}{method}-{output_name}', gamma)
