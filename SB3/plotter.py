# Optuna Plotter:

from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_param_importances
import plotly

#SB3 Plotter
from stable_baselines3.common.results_plotter import load_results, ts2xy
import numpy as np
import matplotlib.pyplot as plt

def Plotter(study, imgDir):
    fig1 = plot_parallel_coordinate(study)
    fig2 = plot_optimization_history(study)
    fig3 = plot_param_importances(study)


    fig1.write_image(imgDir +"plot_parallel_coordinate.jpeg")
    fig2.write_image(imgDir +"plot_optimization_history.jpeg")
    fig3.write_image(imgDir +"plot_param_importances.jpeg")


def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')


def plot_results(log_folder, img_name, img_folder, maxlengthpercentage=1, window_size=50, title='Learning Curve'):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    folder_image_name = str(img_folder + img_name)

    x, y = ts2xy(load_results(log_folder), 'timesteps')

    maxlength = int((np.shape(x)[0])*maxlengthpercentage)

    x = x[int(0):int(maxlength)]
    y = y[int(0):int(maxlength)]

    y = moving_average(y, window=window_size)

    x = x[len(x) - len(y):]

    fig = plt.figure(title, figsize=[5.0,4.0])

    scale = np.amax(x)/5

    #specify x-axis locations
    x_ticks = [0, round(1*scale, -3), round(2*scale, -3), round(3*scale,-3), round(4*scale, -3), round(5*scale, -3)]
    # print(x_ticks)

    #add x-axis values to plot
    plt.xticks(ticks=x_ticks)
    plt.plot(x, y, linestyle='solid')
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.title(title + " Smoothed")
    plt.savefig(folder_image_name)
    plt.close()