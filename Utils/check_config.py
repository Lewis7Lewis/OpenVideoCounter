"""Module to check the config and drow the autorized paths"""
import matplotlib.pyplot as plt
import numpy as np
from anayzer import Analyser


if __name__ == "__main__":
    analys = Analyser("config.toml")
    analys.open()
    print(analys.config)
    shape = np.array([640, 820])  # width,heigh

    midle = shape // 2
    fig = plt.figure("Travels graph")
    plt.axline(
        (0, analys.config["fence"]["l"] * shape[1] / 100),
        (shape[0], shape[1] * analys.config["fence"]["r"] / 100),
    )
    for angle in np.linspace(0, np.pi * 2, 4 * 6, endpoint=False):
        trajet = np.array([[np.cos(angle), np.sin(angle)], [-np.cos(angle), -np.sin(angle)]])
        trajet *= analys.config["filters"]["minWalkingDistance"]
        trajet += midle
        passing, er = analys.passing_fences(trajet, np.concat([shape[::-1], [0]]))
        print("state :", passing, er, "points :", trajet, "vector :", (trajet[1] - trajet[0]))

        if passing:
            plt.quiver(
                *trajet[0],
                *(trajet[1] - trajet[0]),
                color="g",
                scale=0.2,
                angles="xy",
                scale_units="xy"
            )
        else:
            plt.quiver(
                *trajet[0],
                *(trajet[1] - trajet[0]),
                color="r",
                scale=0.2,
                angles="xy",
                scale_units="xy"
            )
    # plt.axis('equal')
    plt.xlim(0, shape[0])
    plt.ylim(shape[1], 0)
    plt.show()
