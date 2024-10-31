import numpy as np
import os
import re
import vtk
import shutil
import time
import json
from collections import defaultdict
from multiprocessing import Pool
import matplotlib
import matplotlib.pyplot as plt
# non-interactive backend
matplotlib.use("Agg")


def extract_cells(connectivity_matrix):
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


def build_segments(cells):
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


def segments_to_dict(segments):
    segments_dict = {}

    for i, segment in enumerate(segments):
        # Flatten the segment to get all IDs in a single list
        flat_segment = [item for sublist in segment for item in sublist]
        # Get the unique IDs in the segment
        unique_ids = sorted(set(flat_segment))
        # Store in the dictionary
        segments_dict[i] = unique_ids

    return segments_dict


def segments_to_location_dict(segments_dict, points):
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
def captureVTKDipolePositions(filename):

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

    return posID_dict


# Define a function to extract the number from the string
def extractNumber(s: str) -> int:
    # return int(re.search(r"\d+", s).group())
    match = re.search(r'quadrature_(\d+)', s)
    # Return a large number if no match is found
    return int(match.group(1)) if match else float('inf')


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
    for partialType, pos in glidePlane.items():
        avgPos[partialType] = np.mean(pos[:, glideDirIdx])
    # calculate the avg separation distance
    avgPos["avgSep"] = np.abs(avgPos["lead"] - avgPos["trail"])
    return avgPos


def writeAvgData(
    #vtkDataDir: str, dataDir: str, avgPos: dict, tStep: int, idx: int, fName: str
    vtkDataDir: str, dataDir: str, avgPos: dict, tStep: int, fName: str
) -> None:
    # create directory if it doesn't exist
    # if not os.path.exists(f"{vtkDataDir}/{dataDir}"):
    #     os.makedirs(f"{vtkDataDir}/{dataDir}")

    # write header
    # if not os.path.exists(f"{vtkDataDir}/{dataDir}/{fName}"):
    #     with open(f"{vtkDataDir}/{dataDir}/{fName}", "a") as f:
    #         f.write(f"# tStep leadAvgPos trailAvgPos sepAvg\n")

    # write the average separation data to a file
    with open(f"{vtkDataDir}/{dataDir}/{fName}", "a") as f:
        f.write(f"{tStep} {avgPos['lead']} {avgPos['trail']} {avgPos['avgSep']}\n")


def plotGlidePlane(
    dislocationType: str, vtkDataDir: str, figDir: str, glidePlane: dict, avgPos: dict, tStep: int
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

    # create plot object
    # fig, ax = plt.subplots(figsize=(6, 13), dpi=200)

    # set color
    color = {"lead": "#b59461", "trail": "#d03a77"}

    for partialType, pos in glidePlane.items():
        # sort the data based on the line tangent direction
        match dislocationType:
            case "edge":
                # create plot object
                sortIdx = np.argsort(pos[:, 1])
                ax.axvline(avgPos[partialType], color=color[partialType], linestyle="--")
            case "screw":
                # create plot object
                sortIdx = np.argsort(pos[:, 0])
                ax.axhline(avgPos[partialType], color=color[partialType], linestyle="--")
            case _:
                exit("invalid dislocation type")
        # sortIdx = np.argsort(pos[:, 1])
        pos = pos[sortIdx]
        # plot dislocation line
        ax.plot(pos[:, 0], pos[:, 1], color=color[partialType])
        # plot the average position
        # ax.axvline(avgPos[partialType], color=color[partialType], linestyle="--")
        # ax.axhline(avgPos[partialType], color=color[partialType], linestyle="--")

    # create directory if it doesn't exist
    #if not os.path.exists(f"{vtkDataDir}/{figDir}"):
    #    os.makedirs(f"{vtkDataDir}/{figDir}")

    ax.grid()
    ax.set_xlabel("x [$\\vec{b}$]")
    ax.set_ylabel("y [$\\vec{b}$]")
    # ax.set_xlabel("x b")
    # ax.set_ylabel("y b")
    figName = f"{vtkDataDir}/{figDir}/{tStep}.png"
    # prevent the labels from being cut off
    plt.tight_layout()
    fig.savefig(figName, transparent=True)
    plt.close()


def pre_render() -> None:
    # Create a dummy plot to force cache building
    fig, ax = plt.subplots()
    ax.text(0.5, 0.5, r"$x^2 + y^2 = z^2$", fontsize=12)
    plt.close()


def process_vtk_file(vtkFilePath: str) -> None:
    # Parse JSON file
    with open("config.json") as f:
        config = json.load(f)

    dislocType = config["dislocationType"]
    # vtkDataDir = config["vtkDataDir"]
    plotDislocation = config["plotDislocation"]
    glidePlaneNormal = config["glidePlaneNormal"]
    dataDir = config["dataDir"]
    glidePlane1PlotDir = config["glidePlane1PlotDir"]
    glidePlane2PlotDir = config["glidePlane2PlotDir"]

    vtkFile = vtkFilePath.split("/")[-1]
    #vtkFilePath = vtkFilePath.split("/")[0:-1]
    vtkDataDir = "/".join(vtkFilePath.split("/")[0:-1])

    # Extract the time step number
    tStep = extractNumber(vtkFile)

    # Capture VTK dipole positions
    # dislocLoops = captureVTKDipolePositions(f"{vtkDataDir}/{vtkFilePath}")
    dislocLoops = captureVTKDipolePositions(f"{vtkFilePath}")
    # print(f"Processing time step: {tStep}, the number of loops: {len(dislocLoops)}")

    # if the number of loops are less than 4
    # stop the execution
    if len(dislocLoops) < 4:
        return

    # axis index dictionary
    axisIdx = {"x": 0, "y": 1, "z": 2}

    # find the loop indexes that have
    # the same glide plane normal coords
    planeNormVals = []
    for idx, val in dislocLoops.items():
        posIdx = axisIdx[glidePlaneNormal]
        val = np.array(val)
        planeNormVals.append((idx, round(np.mean(val[:,posIdx]))))

    # group the indexes based on the glide plane number
    loopIdx = {}
    for idx1, vals1  in planeNormVals:
        for idx2, vals2 in planeNormVals:
            if vals1 == vals2 and idx1 != idx2:
                if len(loopIdx) == 0:
                    loopIdx["plane1"] = np.sort([idx1, idx2])
                else:
                    loopIdx["plane2"] = np.sort([idx1, idx2])
                    break

    p1Idx = loopIdx["plane1"]
    glidePlane1 = {
        "lead": np.array(dislocLoops[p1Idx[0]]),
        "trail": np.array(dislocLoops[p1Idx[1]]),
    }
    p2Idx = loopIdx["plane2"]
    glidePlane2 = {
        "lead": np.array(dislocLoops[p2Idx[0]]),
        "trail": np.array(dislocLoops[p2Idx[1]]),
    }

    #glidePlane1 = {
    #    "lead": np.array(dislocLoops[0]),
    #    "trail": np.array(dislocLoops[2]),
    #}

    #glidePlane2 = {
    #    "lead": np.array(dislocLoops[1]),
    #    "trail": np.array(dislocLoops[3]),
    #}

    # Calculate leading/trailing avg position/avg separation distance
    gP1AvgPos = calculateAvgPos(dislocType, glidePlane1)
    gP2AvgPos = calculateAvgPos(dislocType, glidePlane2)

    # Write average data
    writeAvgData(vtkDataDir, dataDir, gP1AvgPos, tStep, "glidePlane1Sep.txt")
    writeAvgData(vtkDataDir, dataDir, gP2AvgPos, tStep, "glidePlane2Sep.txt")

    # create figure directory if it doesn't exist
    # if not os.path.exists(f"{vtkDataDir}/{glidePlane1PlotDir}"):
    #     os.makedirs(f"{vtkDataDir}/{glidePlane1PlotDir}")
    # if not os.path.exists(f"{vtkDataDir}/{glidePlane2PlotDir}"):
    #     os.makedirs(f"{vtkDataDir}/{glidePlane2PlotDir}")

    if plotDislocation:
        plotGlidePlane(dislocType, vtkDataDir, glidePlane1PlotDir, glidePlane1, gP1AvgPos, tStep)
        plotGlidePlane(dislocType, vtkDataDir, glidePlane2PlotDir, glidePlane2, gP2AvgPos, tStep)


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
    #vtkDataDir = config["vtkDataDir"]
    plotDislocation = config["plotDislocation"]
    dataDir = config["dataDir"]
    glidePlane1PlotDir = config["glidePlane1PlotDir"]
    glidePlane2PlotDir = config["glidePlane2PlotDir"]

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

        # Read files
        vtkDislocFiles = []
        with os.scandir(vtkDataDir) as entries:
            for entry in entries:
                if re.search(r"quadrature_", entry.name) and entry.is_file():
                    #vtkDislocFiles.append(entry.name)
                    vtkDislocFiles.append(entry.path)

        # Sort the data based on the time step number
        vtkDislocFiles = sorted(vtkDislocFiles, key=extractNumber)

        # create data/figure directory
        os.makedirs(f"{vtkDataDir}/{dataDir}")
        os.makedirs(f"{vtkDataDir}/{glidePlane1PlotDir}")
        os.makedirs(f"{vtkDataDir}/{glidePlane2PlotDir}")

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

        startT = time.perf_counter()
        # process the data with multiple threads
        with Pool(processes=workers) as pool:
            results = pool.map(process_vtk_file, vtkDislocFiles)
            # exectue in order
        endT = time.perf_counter()
        duration = endT - startT
        print(f"{seedDataDir} prcessing task took {duration:.2f}s total")

    return 0


if __name__ == "__main__":
    main()
