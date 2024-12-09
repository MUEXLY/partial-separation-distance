import os
import re
import vtk
import shutil
import time
import json
import numpy as np
import matplotlib
import matplotlib.animation as animation

# non-interactive backend
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict
from multiprocessing import Pool


def extract_cells(connectivity_matrix: np.ndarray) -> np.ndarray:
    cells = []
    i = 0
    while i < len(connectivity_matrix):
        # The first element is the number of points in this cell
        num_points = connectivity_matrix[i]
        # The next 'num_points' elements are the points in the cell
        cell = connectivity_matrix[i + 1 : i + 1 + num_points]
        # Append the cell to the list of cells
        cells.append(cell)
        # Move to the next cell
        i += num_points + 1
    return cells


def build_segments(cells: dict) -> list:
    # Dictionary to store connections between cells
    connectivity_map = defaultdict(set)

    # Iterate over each cell
    for i, cell in enumerate(cells):
        for index in cell:
            connectivity_map[index].add(i)

    # Function to perform a Depth-First Search (DFS) on the graph
    def dfs(cell_idx, visited, segment):
        visited.add(cell_idx)
        segment.append(cells[cell_idx])
        # Traverse all the neighbors (connected cells)
        for index in cells[cell_idx]:
            for neighbor in connectivity_map[index]:
                if neighbor not in visited:
                    dfs(neighbor, visited, segment)

    visited = set()
    segments = []

    # Iterate through all cells and find connected components
    for cell_idx in range(len(cells)):
        if cell_idx not in visited:
            segment = []
            dfs(cell_idx, visited, segment)
            segments.append(segment)

    return segments


def segments_to_dict(segments: list) -> dict:
    segments_dict = {}

    for i, segment in enumerate(segments):
        # Flatten the segment to get all IDs in a single list
        flat_segment = [item for sublist in segment for item in sublist]
        # Get the unique IDs in the segment
        unique_ids = sorted(set(flat_segment))
        # Store in the dictionary
        segments_dict[i] = unique_ids

    return segments_dict


def segments_to_location_dict(segments_dict: dict, points: np.ndarray) -> dict:
    segments_location_dict = {}

    for segment_id, point_ids in segments_dict.items():
        # Retrieve the 1x3 vectors for each point ID
        location_vectors = [points[point_id] for point_id in point_ids]
        # Store in the dictionary
        segments_location_dict[segment_id] = location_vectors

    return segments_location_dict


def calculate_average(dictionary):
    average_dictionary = {}

    for key, value in dictionary.items():
        average_value = sum(value) / len(value)
        average_dictionary[key] = average_value

    return average_dictionary


# write a function to capture nodal positions from .vtk and detect outliers
# return: dictionary of positions where the key is the dislocation index
def captureVTKDipolePositions(filename: str):

    reader = vtk.vtkGenericDataObjectReader()  # set the vtkReader
    reader.SetFileName(filename)  # declare the vtk filename
    reader.ReadAllVectorsOn()  # read all vector data
    reader.ReadAllScalarsOn()  # read all scalar data
    reader.Update()  # update to new file

    # CREATES DYNAMIC POINT DATA OBJECT
    pointData = reader.GetOutput().GetPointData()
    points = np.array(
        reader.GetOutput().GetPoints().GetData()
    )  # READS ALL PHYSICAL POINT LOCATIONS
    cells = reader.GetOutput().GetCells().GetData()  # READS ALL CELLS
    connectivity_array = np.array(cells)
    individual_cells = extract_cells(connectivity_array)
    segments = build_segments(individual_cells)
    segmentID_dict = segments_to_dict(segments)
    posID_dict = segments_to_location_dict(segmentID_dict, points)

    # return nodal positions
    return posID_dict

# def segments_to_velocity_dict(segments_dict, velocities):
#     segments_velocity_dict = {}
# 
#     for segment_id, point_ids in segments_dict.items():
#         # Retrieve the 1x3 vectors for each point ID and filter out zero vectors
#         velocity_vectors = [velocities[point_id] for point_id in point_ids if not np.all(velocities[point_id] == 0)]
#         # Store in the dictionary
#         segments_velocity_dict[segment_id] = velocity_vectors
#     
#     return segments_velocity_dict


# Define a function to extract the number from the string
def extractNumber(s: str) -> int:
    # return int(re.search(r"\d+", s).group())
    match = re.search(r"quadrature_(\d+)", s)
    # Return a large number if no match is found
    return int(match.group(1)) if match else float("inf")


def calculateAvgPos(dislocationType: str, glidePlane: dict) -> float:
    # calculate the avg position
    avgPos = {}
    match dislocationType:
        case "edge":
            glideDirIdx = 0
        case "screw":
            glideDirIdx = 1
        case _:
            exit("invalid dislocation type")
    # glideDirIdx = 0
    # loop through each partial type (lead, trail)
    for planeNum, pos in glidePlane.items():
        avgPos[planeNum] = np.mean(pos[:, glideDirIdx])
    # calculate the avg separation distance
    #avgPos["avgSep"] = np.abs(avgPos["lead"] - avgPos["trail"])
    return avgPos


def writeAvgData(
    vtkDataDir: str,
    dataDir: str,
    avgPos: dict,
    tStep: int,
    fName: str,
) -> None:
    for planeNum, avgVal in avgPos.items():
        match planeNum:
            case 'plane1':
                plane = 'plane1'
            case 'plane2':
                plane = 'plane2'
    # write the average separation data to a file
    with open(f"{vtkDataDir}/{dataDir}/{fName}", "a") as f:
        f.write(f"{tStep} {avgPos[plane]}\n")


def plotBothGlidePlane(
    dislocationType: str, vtkDataDir: str, figDir: str,
    glidePlane1: dict, glidePlane2: dict,
    avgPos1: dict, avgPos2: dict,
    tStep: int,
) -> None:
    match dislocationType:
        case "edge":
            # create plot object
            #fig, ax = plt.subplots(1, 2, figsize=(6, 15), dpi=200)
            fig, ax = plt.subplots(1, 2, figsize=(9, 13), dpi=200)
        case "screw":
            # create plot object
            fig, ax = plt.subplots(2, 1, figsize=(13, 9), dpi=200)
        case _:
            exit("invalid dislocation type")
    # set color
    color = {"plane1": "#b59461", "plane2": "#d03a77"}

    for partialType, pos in glidePlane1.items():
        match dislocationType:
            case "edge":
                # create an array of sorting index based on the line tangent direction
                sortIdx = np.argsort(pos[:, 1])
                # draw average line position
                ax[0].axvline(
                    avgPos1[partialType], color=color[partialType], linestyle="--"
                )
            case "screw":
                # create an array of sorting index based on the line tangent direction
                sortIdx = np.argsort(pos[:, 0])
                # draw average line position
                ax[0].axhline(
                    avgPos1[partialType], color=color[partialType], linestyle="--"
                )
            case _:
                exit("invalid dislocation type")

        # sort the data based on the line tangent direction
        pos = pos[sortIdx]
        # plot dislocation line
        # ax.plot(pos[:, 0], pos[:, 1], '-o', markersize=0.8, color=color[partialType])
        ax[0].plot(pos[:, 0], pos[:, 1], color=color[partialType])

    for partialType, pos in glidePlane2.items():
        match dislocationType:
            case "edge":
                # create an array of sorting index based on the line tangent direction
                sortIdx = np.argsort(pos[:, 1])
                # draw average line position
                ax[1].axvline(
                    avgPos2[partialType], color=color[partialType], linestyle="--"
                )
            case "screw":
                # create an array of sorting index based on the line tangent direction
                sortIdx = np.argsort(pos[:, 0])
                # draw average line position
                ax[1].axhline(
                    avgPos2[partialType], color=color[partialType], linestyle="--"
                )
            case _:
                exit("invalid dislocation type")

        # sort the data based on the line tangent direction
        pos = pos[sortIdx]
        # plot dislocation line
        # ax.plot(pos[:, 0], pos[:, 1], '-o', markersize=0.8, color=color[partialType])
        ax[1].plot(pos[:, 0], pos[:, 1], color=color[partialType])

    ax[0].grid()
    ax[0].set_xlabel("x [$\\AA$]")
    ax[0].set_ylabel("y [$\\AA$]")
    ax[0].set(title="plane 1")

    ax[1].grid()
    ax[1].set_xlabel("x [$\\AA$]")
    ax[1].set_ylabel("y [$\\AA$]")
    ax[1].set(title="plane 2")

    # ax.set_xlabel("x b")
    # ax.set_ylabel("y b")
    figName = f"{vtkDataDir}/{figDir}/{tStep}.png"
    # prevent the labels from being cut off
    plt.tight_layout()
    fig.savefig(figName, transparent=False)
    plt.close()



def plotGlidePlane(
    dislocationType: str,
    vtkDataDir: str,
    figDir: str,
    glidePlane: dict,
    avgPos: dict,
    tStep: int,
) -> None:
    match dislocationType:
        case "edge":
            # create plot object
            fig, ax = plt.subplots(figsize=(6, 13), dpi=200)
        case "screw":
            # create plot object
            fig, ax = plt.subplots(figsize=(13, 6), dpi=200)
        case _:
            exit("invalid dislocation type")

    # set color
    color = {"plane1": "#b59461", "plane2": "#d03a77"}

    for partialType, pos in glidePlane.items():
        match dislocationType:
            case "edge":
                # create an array of sorting index based on the line tangent direction
                sortIdx = np.argsort(pos[:, 1])
                # draw average line position
                ax.axvline(
                    avgPos[partialType], color=color[partialType], linestyle="--"
                )
            case "screw":
                # create an array of sorting index based on the line tangent direction
                sortIdx = np.argsort(pos[:, 0])
                # draw average line position
                ax.axhline(
                    avgPos[partialType], color=color[partialType], linestyle="--"
                )
            case _:
                exit("invalid dislocation type")
        # sort the data based on the line tangent direction
        pos = pos[sortIdx]
        # plot dislocation line
        # ax.plot(pos[:, 0], pos[:, 1], '-o', markersize=0.8, color=color[partialType])
        ax.plot(pos[:, 0], pos[:, 1], color=color[partialType])

    ax.grid()
    ax.set_xlabel("x [$\\AA$]")
    ax.set_ylabel("y [$\\AA$]")
    # ax.set_xlabel("x b")
    # ax.set_ylabel("y b")
    figName = f"{vtkDataDir}/{figDir}/{tStep}.png"
    # prevent the labels from being cut off
    plt.tight_layout()
    fig.savefig(figName, transparent=False)
    plt.close()


def pre_render() -> None:
    # Create a dummy plot to force cache building
    fig, ax = plt.subplots()
    ax.text(0.5, 0.5, r"$x^2 + y^2 = z^2$", fontsize=12)
    plt.close()


def plotSepationDist(
    dislocationType: str,
    vtkDataDir: str,
    figDir: str,
    glidePlane: dict,
    tStep: int,
) -> None:
    # glidePlane dict has "lead" and "trail" keys
    for partialType, pos in glidePlane.items():
        match dislocationType:
            case "edge":
                # create an array of sorting index based on the line tangent direction
                sortIdx = np.argsort(pos[:, 1])
                # draw average line position
            case "screw":
                # create an array of sorting index based on the line tangent direction
                sortIdx = np.argsort(pos[:, 0])
                # draw average line position
            case _:
                exit("invalid dislocation type")
        # sort the data based on the line tangent direction
        pos = pos[sortIdx]
        glidePlane[partialType] = np.around(pos, decimals=3)
        # plot dislocation line
        # ax.plot(pos[:, 0], pos[:, 1], '-o', markersize=0.8, color=color[partialType])
        # ax.plot(pos[:, 0], pos[:, 1], color=color[partialType])
    xIdx, yIdx = 0, 1
    separationDists = []

    for nodePosLead in glidePlane["lead"]:
        for nodePosTrail in glidePlane["trail"]:
            match dislocationType:
                case "edge":
                    if np.abs(nodePosLead[yIdx] - nodePosTrail[yIdx]) < 0.1:
                        nodeSeperationDist = np.abs(nodePosLead[xIdx]-nodePosTrail[xIdx])
                        separationDists.append(nodeSeperationDist)
                        break
                case "screw":
                    if np.abs(nodePosLead[xIdx] - nodePosTrail[xIdx]) < 0.1:
                        nodeSeperationDist = np.abs(nodePosLead[yIdx]-nodePosTrail[yIdx])
                        separationDists.append(nodeSeperationDist)
                        break
                case _:
                    exit("invalid dislocation type")
    separationDists = np.array(separationDists)

    # save data
    np.savetxt(f"{vtkDataDir}/{figDir}/{tStep}.txt", separationDists)

    # create plot object
    #fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    #ax.hist(separationDists)
    #figName = f"{vtkDataDir}/{figDir}/{tStep}.png"
    #fig.savefig(figName)
    #plt.close()


def renderVideo(
    vtkDataDir: str,
    glidePlane1PlotDir: str,
    glidePlane2PlotDir: str,
    dislocationType: str,
    results: list,
) -> None:
    match dislocationType:
        case "edge":
            # create plot object
            fig, ax = plt.subplots(figsize=(6, 13), dpi=200)
            # determine the plot range
            tStep, lastGlidePlane1, lastGlidePlane2 = results[-1]
            tStep, firstGlidePlane1, firstGlidePlane2 = results[0]
            glideDirMaxLead = np.max(lastGlidePlane1["lead"][:, 0])
            glideDirMinTrail = np.min(firstGlidePlane1["trail"][:, 0])
        case "screw":
            # create plot object
            fig, ax = plt.subplots(figsize=(13, 6), dpi=200)
            # determine the plot range, second column is y dir (glidedir)
            tStep, lastGlidePlane1, lastGlidePlane2 = results[-1]
            tStep, firstGlidePlane1, firstGlidePlane2 = results[0]
            glideDirMaxLead = np.max(lastGlidePlane1["lead"][:, 1])
            glideDirMinTrail = np.min(firstGlidePlane1["trail"][:, 1])
        case _:
            exit("invalid dislocation type")
    plt.tight_layout()
    ax.set_xlabel("")
    ax.set_ylabel("")
    buffer = 3
    ax.set_xlim(glideDirMinTrail - buffer, glideDirMaxLead + buffer)
    ax.grid(True)

    imgs = []
    for loopIdx, result in enumerate(results):
        tStep = result[0]
        glidePlane = result[1]
        # glidePlane2 = result[2]
        # set color
        color = {"lead": "#b59461", "trail": "#d03a77"}
        for idx, (partialType, pos) in enumerate(glidePlane.items()):
            match dislocationType:
                case "edge":
                    # create an array of sorting index based on the line tangent direction
                    sortIdx = np.argsort(pos[:, 1])
                case "screw":
                    # create an array of sorting index based on the line tangent direction
                    sortIdx = np.argsort(pos[:, 0])
                case _:
                    exit("invalid dislocation type")
            # sort the data based on the line tangent direction
            pos = pos[sortIdx]

            if idx == 0:
                img1 = ax.plot(
                    pos[:, 0], pos[:, 1], color=color[partialType], animated=True
                )[0]
            else:
                # animation function can only take the list of matplotlib object, so [0]
                img2 = ax.plot(
                    pos[:, 0], pos[:, 1], color=color[partialType], animated=True
                )[0]

        imgs.append([img1, img2])

    ani = animation.ArtistAnimation(
        fig, imgs, interval=50, blit=True, repeat_delay=1000
    )

    writer = animation.FFMpegWriter(
        fps=15, metadata=dict(artist="Me"), bitrate=1800, codec="vp9"
    )

    ani.save(f"{vtkDataDir}/{glidePlane1PlotDir}/video1.webm", writer=writer)
    plt.close()

    match dislocationType:
        case "edge":
            # create plot object
            fig2, ax2 = plt.subplots(figsize=(6, 13), dpi=200)
            # determine the plot range
            tStep, lastGlidePlane1, lastGlidePlane2 = results[-1]
            tStep, firstGlidePlane1, firstGlidePlane2 = results[0]
            glideDirMinLead = np.min(lastGlidePlane2["lead"][:, 0])
            glideDirMaxTrail = np.max(firstGlidePlane2["trail"][:, 0])
        case "screw":
            # create plot object
            fig2, ax2 = plt.subplots(figsize=(13, 6), dpi=200)
            # determine the plot range, second column is y dir (glidedir)
            tStep, lastGlidePlane1, lastGlidePlane2 = results[-1]
            tStep, firstGlidePlane1, firstGlidePlane2 = results[0]
            glideDirMinLead = np.min(lastGlidePlane2["lead"][:, 1])
            glideDirMaxTrail = np.max(firstGlidePlane2["trail"][:, 1])
        case _:
            exit("invalid dislocation type")
    plt.tight_layout()
    ax2.set_xlabel("")
    ax2.set_ylabel("")
    buffer = 3
    ax2.set_xlim(glideDirMinLead - buffer, glideDirMaxTrail + buffer)
    ax2.grid(True)
    imgs = []
    for loopIdx, result in enumerate(results):
        tStep = result[0]
        glidePlane = result[2]
        # glidePlane2 = result[2]

        # set color
        color = {"lead": "#b59461", "trail": "#d03a77"}
        for idx, (partialType, pos) in enumerate(glidePlane.items()):
            match dislocationType:
                case "edge":
                    # create an array of sorting index based on the line tangent direction
                    sortIdx = np.argsort(pos[:, 1])
                case "screw":
                    # create an array of sorting index based on the line tangent direction
                    sortIdx = np.argsort(pos[:, 0])
                case _:
                    exit("invalid dislocation type")
            # sort the data based on the line tangent direction
            pos = pos[sortIdx]

            if idx == 0:
                img1 = ax2.plot(
                    pos[:, 0], pos[:, 1], color=color[partialType], animated=True
                )[0]
            else:
                # animation function can only take the list of matplotlib object, so [0]
                img2 = ax2.plot(
                    pos[:, 0], pos[:, 1], color=color[partialType], animated=True
                )[0]

        imgs.append([img1, img2])

    ani = animation.ArtistAnimation(
        fig2, imgs, interval=50, blit=True, repeat_delay=1000
    )

    writer = animation.FFMpegWriter(
        fps=15, metadata=dict(artist="Me"), bitrate=1800, codec="vp9"
    )

    ani.save(f"{vtkDataDir}/{glidePlane2PlotDir}/video2.webm", writer=writer)


def plotTotalDistribution(
    vtkDataDir: str,
    totalPlane1DistDir: str,
    totalPlane2DistDir: str,
    dislocationType: str,
    results: list,
) -> None:

    totalSepDists = []
    for loopIdx, result in enumerate(results):
        tStep = result[0]

        # wait until the dislocations find the stable state
        if tStep < 500:
            continue

        glidePlane1 = result[1]
        glidePlane2 = result[2]

        #for idx, (partialType, pos) in enumerate(glidePlane1.items()):
        for partialType, pos in glidePlane1.items():
            match dislocationType:
                case "edge":
                    # create an array of sorting index based on the line tangent direction
                    sortIdx = np.argsort(pos[:, 1])
                case "screw":
                    # create an array of sorting index based on the line tangent direction
                    sortIdx = np.argsort(pos[:, 0])
                case _:
                    exit("invalid dislocation type")
            # sort the data based on the line tangent direction
            pos = pos[sortIdx]
            glidePlane1[partialType] = np.around(pos, decimals=3)

        xIdx, yIdx = 0, 1
        separationDists = []
        for nodePosLead in glidePlane1["lead"]:
            for nodePosTrail in glidePlane1["trail"]:
                match dislocationType:
                    case "edge":
                        if np.abs(nodePosLead[yIdx] - nodePosTrail[yIdx]) < 0.1:
                            nodeSeperationDist = np.abs(nodePosLead[xIdx]-nodePosTrail[xIdx])
                            separationDists.append(nodeSeperationDist)
                            break
                    case "screw":
                        if np.abs(nodePosLead[xIdx] - nodePosTrail[xIdx]) < 0.1:
                            nodeSeperationDist = np.abs(nodePosLead[yIdx]-nodePosTrail[yIdx])
                            separationDists.append(nodeSeperationDist)
                            break
                    case _:
                        exit("invalid dislocation type")
        separationDists = np.array(separationDists)
        totalSepDists.append(separationDists)

    totalSepDists = np.array(totalSepDists)
    # create plot object
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    ax.hist(totalSepDists)
    figName = f"{vtkDataDir}/{totalPlane1DistDir}/separationDistribution.png"
    fig.savefig(figName)
    plt.close()


def process_vtk_file(vtkFilePath: str) -> None:
    # Parse JSON file
    with open("config.json") as f:
        config = json.load(f)

    # find alloy type
    pattern = r"AlMg\d+"
    for j in vtkFilePath.split('/'):
        if 'AlMg' in j:
            dummy = j.split('-')
            for dummyStr in dummy:
                alloy = re.search(pattern, dummyStr)
                if alloy:
                    alloy = alloy.group(0)
                    break

    # vtkFilePath includes the root vtk file directory
    for i in vtkFilePath.split('/'):
        # find the string that has the pattern 'AlMg5-IMG1-E/S'
        if f"{alloy}-" in i:
            # match a letter either E or S
            letterPattern = r'[ES]'
            #disType = i.split('-')[-1]
            disType = [ x for x in i.split('-') if re.search(letterPattern, x) ]
            disType = ''.join(disType[0])
            match disType:
                case "E":
                    dislocType = "edge"
                case "S":
                    dislocType = "screw"

    # dislocType = config["dislocationType"]
    # vtkDataDir = config["vtkDataDir"]
    plotDislocation = config["plotDislocation"]
    glidePlaneNormal = config["glidePlaneNormal"]
    dataDir = config["dataDir"]
    glidePlane1PlotDir = config["glidePlane1PlotDir"]
    glidePlane2PlotDir = config["glidePlane2PlotDir"]
    glidePlaneBothPlotDir = config["glidePlaneBothPlotDir"]
    glidePlane1DistDir = config["glidePlane1DistDir"]
    glidePlane2DistDir = config["glidePlane2DistDir"]
    saveSeparationDist = config["saveSeparationDist"]
    onlyLastStep = config["processOnlyLastStep"]

    vtkFile = vtkFilePath.split("/")[-1]

    # vtkFilePath = vtkFilePath.split("/")[0:-1]
    vtkDataDir = "/".join(vtkFilePath.split("/")[0:-1])

    # Extract the time step number
    tStep = extractNumber(vtkFile)

    # Capture VTK dipole positions
    # dislocLoops = captureVTKDipolePositions(f"{vtkDataDir}/{vtkFilePath}")
    dislocLoops = captureVTKDipolePositions(f"{vtkFilePath}")
    # print(f"Processing time step: {tStep}, the number of loops: {len(dislocLoops)}")

    # if the number of loops are less than 4
    # stop the execution
    #if len(dislocLoops) < 4:
    #    return

    # axis index dictionary
    axisIdx = {"x": 0, "y": 1, "z": 2}

    # find the loop indexes that have
    # the same glide plane normal coords
    planeNormVals = []
    for idx, val in dislocLoops.items():
        posIdx = axisIdx[glidePlaneNormal]
        val = np.array(val)
        planeNormVals.append((idx, round(np.mean(val[:, posIdx]))))

    # group the indexes based on the glide plane number
    loopIdx = {}
    for loopI, (idx1, vals1) in enumerate(planeNormVals):
        if loopI == 0:
            previousVal = vals1
            loopIdx["plane1"] = idx1
        else:
            if vals1 != previousVal:
                loopIdx["plane2"] = idx1

    glidePlane1 = {
        "plane1": np.array(dislocLoops[loopIdx["plane1"]]),
    }
    glidePlane2 = {
        "plane2": np.array(dislocLoops[loopIdx["plane2"]]),
    }

    # Calculate leading/trailing avg position/avg separation distance
    gP1AvgPos = calculateAvgPos(dislocType, glidePlane1)
    gP2AvgPos = calculateAvgPos(dislocType, glidePlane2)

    # Write average data
    writeAvgData(vtkDataDir, dataDir, gP1AvgPos, tStep, "glidePlane1avg.txt")
    writeAvgData(vtkDataDir, dataDir, gP2AvgPos, tStep, "glidePlane2avg.txt")

    if plotDislocation:
        plotGlidePlane(
            dislocType, vtkDataDir, glidePlane1PlotDir, glidePlane1, gP1AvgPos, tStep
        )
        plotGlidePlane(
            dislocType, vtkDataDir, glidePlane2PlotDir, glidePlane2, gP2AvgPos, tStep
        )
        plotBothGlidePlane(
            dislocType, vtkDataDir, glidePlaneBothPlotDir,
            glidePlane1, glidePlane2, gP1AvgPos, gP2AvgPos,
            tStep
        )

    if saveSeparationDist:
        plotSepationDist(
            dislocType, vtkDataDir, glidePlane1DistDir, glidePlane1, tStep
        )
        plotSepationDist(
            dislocType, vtkDataDir, glidePlane2DistDir, glidePlane2, tStep
        )

    return tStep, glidePlane1, glidePlane2


def main():
    # Define default plot properties
    font = {"family": "serif", "weight": "normal", "size": 17}
    mathfont = {"fontset": "stix"}
    plt.rc("font", **font)
    plt.rc("mathtext", **mathfont)

    # Create a dummy plot to force cache building
    pre_render()

    # Parse JSON file
    with open("config.json") as f:
        config = json.load(f)

    rootDir = config["rootDir"]

    # find alloy type
    pattern = r"AlMg\d+"
    for j in rootDir.split('/'):
        if 'AlMg' in j:
            dummy = j.split('-')
            for dummyStr in dummy:
                alloy = re.search(pattern, dummyStr)
                if alloy:
                    alloy = alloy.group(0)
                    break

    # determine dislocation type
    # vtkFilePath includes the root vtk file directory
    for i in rootDir.split("/"):
        # find the string that has the pattern 'AlMg5-IMG1-E/S'
        if f"{alloy}-" in i:
            disType = i.split("-")[-1]
            match disType:
                case "E":
                    dislocType = "edge"
                case "S":
                    dislocType = "screw"
    #dislocType = config["dislocationType"]

    # vtkDataDir = config["vtkDataDir"]
    plotDislocation = config["plotDislocation"]
    plotTotalSepDist = config["plotTotalSepDist"]
    dataDir = config["dataDir"]
    glidePlane1PlotDir = config["glidePlane1PlotDir"]
    glidePlane2PlotDir = config["glidePlane2PlotDir"]
    glidePlaneBothPlotDir = config["glidePlaneBothPlotDir"]
    glidePlane1DistDir = config["glidePlane1DistDir"]
    glidePlane2DistDir = config["glidePlane2DistDir"]
    totalPlane1DistDir = config["totalPlane1DistDir"]
    totalPlane2DistDir = config["totalPlane2DistDir"]
    onlyLastStep = config["processOnlyLastStep"]
    saveVideo = config["saveVideo"]

    seedDataDirs = []
    with os.scandir(rootDir) as dirs:
        for dir in dirs:
            if dir.is_dir() and re.search(r"Str", dir.name):
                seedDataDirs.append(dir.name)


    for seedDataDir in seedDataDirs:
        print(f"processing {seedDataDir}...")
        vtkDataDir = f"{rootDir}/{seedDataDir}/"

        # Clean up the old data
        if os.path.exists(f"{vtkDataDir}/{dataDir}"):
            shutil.rmtree(f"{vtkDataDir}/{dataDir}", ignore_errors=True)
        if os.path.exists(f"{vtkDataDir}/{glidePlane1PlotDir}"):
            shutil.rmtree(f"{vtkDataDir}/{glidePlane1PlotDir}", ignore_errors=True)
        if os.path.exists(f"{vtkDataDir}/{glidePlane2PlotDir}"):
            shutil.rmtree(f"{vtkDataDir}/{glidePlane2PlotDir}", ignore_errors=True)
        if os.path.exists(f"{vtkDataDir}/{glidePlaneBothPlotDir}"):
            shutil.rmtree(f"{vtkDataDir}/{glidePlaneBothPlotDir}", ignore_errors=True)
        if os.path.exists(f"{vtkDataDir}/{glidePlane1DistDir}"):
            shutil.rmtree(f"{vtkDataDir}/{glidePlane1DistDir}", ignore_errors=True)
        if os.path.exists(f"{vtkDataDir}/{glidePlane2DistDir}"):
            shutil.rmtree(f"{vtkDataDir}/{glidePlane2DistDir}", ignore_errors=True)
        if os.path.exists(f"{vtkDataDir}/{totalPlane1DistDir}"):
            shutil.rmtree(f"{vtkDataDir}/{totalPlane1DistDir}", ignore_errors=True)
        if os.path.exists(f"{vtkDataDir}/{totalPlane2DistDir}"):
            shutil.rmtree(f"{vtkDataDir}/{totalPlane2DistDir}", ignore_errors=True)

        # Read files
        vtkDislocFiles = []
        with os.scandir(vtkDataDir) as entries:
            for entry in entries:
                if re.search(r"quadrature_", entry.name) and entry.is_file():
                    # vtkDislocFiles.append(entry.name)
                    vtkDislocFiles.append(entry.path)

        # Sort the data based on the time step number
        vtkDislocFiles = sorted(vtkDislocFiles, key=extractNumber)

        if onlyLastStep:
            # only process the last file
            vtkDislocFiles = [vtkDislocFiles[-1]]

        # create data/figure directory
        os.makedirs(f"{vtkDataDir}/{dataDir}")
        os.makedirs(f"{vtkDataDir}/{glidePlane1PlotDir}")
        os.makedirs(f"{vtkDataDir}/{glidePlane2PlotDir}")
        os.makedirs(f"{vtkDataDir}/{glidePlaneBothPlotDir}")
        os.makedirs(f"{vtkDataDir}/{glidePlane1DistDir}")
        os.makedirs(f"{vtkDataDir}/{glidePlane2DistDir}")
        os.makedirs(f"{vtkDataDir}/{totalPlane1DistDir}")
        os.makedirs(f"{vtkDataDir}/{totalPlane2DistDir}")

        # write header
        if not os.path.exists(f"{vtkDataDir}/{dataDir}/glidePlane1Sep.txt"):
            with open(f"{vtkDataDir}/{dataDir}/glidePlane1Sep.txt", "a") as f:
                f.write(f"# tStep leadAvgPos trailAvgPos sepAvg\n")
        if not os.path.exists(f"{vtkDataDir}/{dataDir}/glidePlane2Sep.txt"):
            with open(f"{vtkDataDir}/{dataDir}/glidePlane2Sep.txt", "a") as f:
                f.write(f"# tStep leadAvgPos trailAvgPos sepAvg\n")

        # get the number of available workers
        # in general, it's not a good idea to use all of
        # available workers, might crash the system
        workers = Pool()._processes - 1
        #workers = 1

        startT = time.perf_counter()
        # process the data with multiple threads
        with Pool(processes=workers) as pool:
            _ = pool.map(process_vtk_file, vtkDislocFiles)
            # exectue in order
            #results = pool.map(process_vtk_file, vtkDislocFiles)
        endT = time.perf_counter()
        duration = endT - startT
        print(f"{seedDataDir} prcessing task took {duration:.2f}s total")

    return 0


if __name__ == "__main__":
    main()
