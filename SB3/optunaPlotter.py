from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_param_importances
import plotly


def Plotter(study, imgDir):
    fig1 = plot_parallel_coordinate(study)
    fig2 = plot_optimization_history(study)
    fig3 = plot_param_importances(study)


    fig1.write_image(imgDir +"plot_parallel_coordinate.jpeg")
    fig2.write_image(imgDir +"plot_optimization_history.jpeg")
    fig3.write_image(imgDir +"plot_param_importances.jpeg")
