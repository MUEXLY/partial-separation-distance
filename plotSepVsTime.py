import os
import re
import json
import numpy as np
import matplotlib.pyplot as plt

bSIalmg5 = 0.286e-9  # [m]
bSIalmg10 = 0.286e-9  # [m]
bSIalmg15 = 0.286e-9  # [m]

mToAng = 1e10
bAlmg5 = bSIalmg5 * mToAng
bAlmg10 = bSIalmg10 * mToAng
bAlmg15 = bSIalmg15 * mToAng


# Define a function to extract the number from the string
def extractNumber(s: str) -> int:
    return int(re.search(r"\d+", s).group())


def main():
    bSIalmg5 = 0.286e-9  # [m]
    bSIalmg10 = 0.286e-9  # [m]
    bSIalmg15 = 0.286e-9  # [m]

    mToAng = 1e10
    bAlmg5 = bSIalmg5 * mToAng
    bAlmg10 = bSIalmg10 * mToAng
    bAlmg15 = bSIalmg15 * mToAng

    dataDir = "./VTKdata/relaxation/no-noise/AlMg5-IMG1-timeIntTest-nodeTest1/edge/Str0/data"

    data = np.loadtxt(f"{dataDir}/glidePlane1Sep.txt", skiprows=1)
    # sort the data based on the timestep
    sortIdx = np.argsort(data[:, 0])
    data = data[sortIdx]
    cutData = data[10:]
    avgSepDist = np.mean(cutData[:, 3])
    fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
    ax.plot(data[:, 0], data[:, 3], color="k", linewidth=2)
    ax.axhline(
        avgSepDist,
        linestyle="--",
        color="#f37735",
        label=f"average distance = {avgSepDist*bAlmg5:.2f} $\\AA$",
    )
    ax.set_title("Glide Plane 1")
    ax.set_xlabel("Step [#]")
    ax.set_ylabel("Separation [$\\vec{b}$]")
    ax.grid()
    ax.legend(frameon=False)
    plt.tight_layout()
    fig.savefig(f"{dataDir}/GP1TimeVsSep.png", transparent=True)
    #plt.show()

    data = np.loadtxt(f"{dataDir}/glidePlane2Sep.txt", skiprows=1)
    # sort the data based on the timestep
    sortIdx = np.argsort(data[:, 0])
    data = data[sortIdx]
    cutData = data[10:]
    avgSepDist = np.mean(cutData[:, 3])
    fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
    ax.plot(data[:, 0], data[:, 3], color="k", linewidth=2)
    ax.axhline(
        avgSepDist,
        linestyle="--",
        color="#f37735",
        label=f"average distance = {avgSepDist*bAlmg5:.2f} $\\AA$",
    )
    ax.set_title("Glide Plane 2")
    ax.set_xlabel("Step [#]")
    ax.set_ylabel("Separation [$\\vec{b}$]")
    ax.grid()
    ax.legend(frameon=False)
    plt.tight_layout()
    fig.savefig(f"{dataDir}/GP2TimeVsSep.png", transparent=True)
    #plt.show()


if __name__ == "__main__":
    main()
