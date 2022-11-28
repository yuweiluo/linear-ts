import numpy as np
import pandas as pd

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




def plot_sample_path( df, x_name,  y_name, x_axis, y_axis,index_name_1, index_name_2,  save_path ,save_name, normalized):
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

                    
                    x = np.arange(horizon) if normalized == 0 else np.arange(horizon)/horizon
                    scale = 2.0
                    lower = mean - scale * se
                    upper = mean + scale * se

                    plt.fill_between(x, lower, upper, alpha=0.2)
                    plt.plot(x, mean, label=index_name_2+'='+str(index_2))


            plt.xlabel(x_axis)
            plt.ylabel(y_axis)
            plt.legend(loc = 'best', prop={'size': 6})
            plt.savefig(save_path+index_name_1+'_'+str(index_1)+'_'+save_name, dpi = 600)


def plot_statistics(figure_folder):
    method_list = ['TS-Bayes_errors','TS-Bayes_lambda_mins', 'TS-Bayes_lambda_maxs','TS-Bayes_log_maxs_over_mins', 'TS-Bayes_thinnesses',  'TS-Bayes_lambdas_second', 'TS-Bayes_lambdas_third', 'TS-Bayes_biases', 'TS-Bayes_variances','TS-Bayes_risks']



    for method in method_list:
        save_path = f'../plots45/{method}/'
        isExist = os.path.exists(save_path)
        if not isExist:
            os.makedirs(save_path)

        df = pd.read_csv(f'../plots/all-last-{name}.csv')
        plot_last_iter(df,'d' ,method,'d','y','k',save_path+name+'-1.jpg' )

        df = pd.read_csv(f'../plots/all-{name}.csv')
        plot_sample_path(df, 'd', method, 't', 'y', 'd', 'k', save_path, f'{name}-2.jpg', 1)
