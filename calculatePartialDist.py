import numpy as np
#import os
#import matplotlib as mpl
import matplotlib.pyplot as plt
#import shutil
#import re
#import pandas as pd
#import math
#import scipy
#from scipy.stats import linregress
#import lammps_logfile
#import endaq #pip3 install -q git+https://github.com/MideTechnology/endaq-python.git@development
#endaq.plot.utilities.set_theme()
import vtk
#from collections import deque
#from collections import defaultdict
#import seaborn as sns
#from scipy.fftpack import fft
#from scipy.stats import maxwell
import json

#def extract_cells(connectivity_matrix):
#    cells = []
#    i = 0
#    while i < len(connectivity_matrix):
#        # The first element is the number of points in this cell
#        num_points = connectivity_matrix[i]
#        # The next 'num_points' elements are the points in the cell
#        cell = connectivity_matrix[i + 1 : i + 1 + num_points]
#        # Append the cell to the list of cells
#        cells.append(cell)
#        # Move to the next cell
#        i += num_points + 1
#    return cells
#
#def build_segments(cells):
#    # Dictionary to store connections between cells
#    connectivity_map = defaultdict(set)
#    
#    # Iterate over each cell
#    for i, cell in enumerate(cells):
#        for index in cell:
#            connectivity_map[index].add(i)
#    
#    # Function to perform a Depth-First Search (DFS) on the graph
#    def dfs(cell_idx, visited, segment):
#        visited.add(cell_idx)
#        segment.append(cells[cell_idx])
#        # Traverse all the neighbors (connected cells)
#        for index in cells[cell_idx]:
#            for neighbor in connectivity_map[index]:
#                if neighbor not in visited:
#                    dfs(neighbor, visited, segment)
#    
#    visited = set()
#    segments = []
#    
#    # Iterate through all cells and find connected components
#    for cell_idx in range(len(cells)):
#        if cell_idx not in visited:
#            segment = []
#            dfs(cell_idx, visited, segment)
#            segments.append(segment)
#    
#    return segments
#
#
#def segments_to_dict(segments):
#    segments_dict = {}
#    
#    for i, segment in enumerate(segments):
#        # Flatten the segment to get all IDs in a single list
#        flat_segment = [item for sublist in segment for item in sublist]
#        # Get the unique IDs in the segment
#        unique_ids = sorted(set(flat_segment))
#        # Store in the dictionary
#        segments_dict[i] = unique_ids
#    
#    return segments_dict
#
#def segments_to_location_dict(segments_dict, points):
#    segments_location_dict = {}
#
#    for segment_id, point_ids in segments_dict.items():
#        # Retrieve the 1x3 vectors for each point ID
#        location_vectors = [points[point_id] for point_id in point_ids]
#        # Store in the dictionary
#        segments_location_dict[segment_id] = location_vectors
#    
#    return segments_location_dict
#
#def calculate_average(dictionary):
#    average_dictionary = {}
#    
#    for key, value in dictionary.items():
#        average_value = sum(value) / len(value)
#        average_dictionary[key] = average_value
#    
#    return average_dictionary
#
##write a function to capture nodal positions from .vtk and detect outliers
##return: dictionary of positions where the key is the dislocation index
#def captureVTKDipolePositions(filename):
#
#    reader = vtk.vtkGenericDataObjectReader() #set the vtkReader
#    reader.SetFileName(filename) #declare the vtk filename
#    reader.ReadAllVectorsOn() #read all vector data
#    reader.ReadAllScalarsOn() #read all scalar data
#    reader.Update() #update to new file
#
#    pointData=reader.GetOutput().GetPointData() #CREATES DYNAMIC POINT DATA OBJECT
#    points = np.array( reader.GetOutput().GetPoints().GetData() )  #READS ALL PHYSICAL POINT LOCATIONS
#    cells=reader.GetOutput().GetCells().GetData() # READS ALL CELLS
#    connectivity_array = np.array(cells)
#    individual_cells=extract_cells(connectivity_array)
#    segments=build_segments(individual_cells)
#    segmentID_dict=segments_to_dict(segments)
#    posID_dict=segments_to_location_dict(segmentID_dict, points)
#
#    return(posID_dict)
#
#def getVTKpos(fname: str):
#    reader = vtk.vtkGenericDataObjectReader() #set the vtkReader
#    reader.SetFileName(fname) #declare the vtk filename
#    reader.Update() #update to new file
#    #reader.ReadAllVectorsOn() #read all vector data
#    reader.ReadAllScalarsOn() #read all scalar data
#    pointData= np.array(reader.GetOutput().GetPoints().GetData())
#    #cells=reader.GetOutput().GetCells().GetData() # READS ALL CELLS
#    b = pointData[:,2]
#    b = np.unique(b)
#    print(b)
#    exit()
#    return 0;


# a = captureVTKDipolePositions('./dislocationVTK/quadrature_0.vtk')
#a = getVTKpos('./dislocationVTK/quadrature_0.vtk')
#print(a)

def main():
    # parse json file
    with open("config.json") as f:
        config = json.load(f)

    glideDirection = config["glideDirection"]
    lineTangent = config["lineTangent"]
    glidePlaneNormal = config["glidePlaneNormal"]

    # axis index dictionary
    axisIdx = {"x":0, "y":1, "z":2}

    file = './dislocationVTK/quadrature_9000.vtk'
    # set the vtkReader
    reader = vtk.vtkGenericDataObjectReader()
    # declare the vtk filename
    reader.SetFileName(file)
    # update to new file
    reader.Update()
    # read all vector datap
    #reader.ReadAllVectorsOn()
    # read all scalar data
    reader.ReadAllScalarsOn()
    # get the point data
    pointData = np.array(reader.GetOutput().GetPoints().GetData(), dtype=np.float16)
    # find the glid plane coordinates
    glidePlanePoints = pointData[:, axisIdx[glidePlaneNormal]]
    glidePlanes = np.unique(glidePlanePoints)

    # go through each glide plane one by one
    for gPlane in glidePlanes:
        # read the point data that has the glide plane coordinate
        planePointData = pointData[glidePlanePoints == gPlane]
        # calculate the average coordinate of the nodes in glide direction
        avg = np.mean(planePointData[:, axisIdx[glideDirection]])
        # nodal position difference tolerance value
        tol = 100
        # remove the zero burgers vector node
        # if the abs nodal pos diff is bigger than tol, remove the node
        planePointData = planePointData[np.abs(planePointData[:, axisIdx[glideDirection]]-avg) < tol]

        # sort the data based on the line tangent direction
        sortIdx = np.argsort(planePointData[:, axisIdx[lineTangent]])
        planePointData = planePointData[sortIdx]

        del(tol)

        # find the leading partial dislocation
        #lead = []
        #b = 2.4
        #xyPos = planePointData[:,0:2]
        #for idx, pos in enumerate(xyPos):
        #    if idx != 0:
        #        if pos[0] != nextPos[0] and pos[1] != nextPos[1]:
        #            continue
        #    lead.append(pos)
        #    for i in xyPos:
        #        v = pos - i
        #        v2 = np.linalg.norm(v)
        #        if v2 < b and v2 != 0:
        #            nextPos = i
        #            lead.append(nextPos)
        #            break
        #lead = np.array(lead)
        #print(lead)
        #exit()
        leadPartial = []
        numOfPoints = len(planePointData)
        xyPos = planePointData[:,0:2]
        yPos = xyPos[:,1]
        for y in yPos:
            # find the row indexes that have the same y coordinate
            rowIdxes = np.argwhere(xyPos == y)[:, 0]
            if len(rowIdxes) < 2:
                continue
            x1, y1 = xyPos[rowIdxes[0]]
            x2, y2 = xyPos[rowIdxes[1]]
            # find the point that has the bigger x coordinate
            leadXpos = np.max([x1, x2])
            if leadXpos == x1:
                leadPoint = xyPos[rowIdxes[0]]
            else:
                leadPoint = xyPos[rowIdxes[1]]
            leadPartial.append(leadPoint)
        leadPartial = np.array(leadPartial)
        print(leadPartial)
        fig, ax = plt.subplots(dpi=100, figsize=(5, 10))
        ax.scatter(leadPartial[:,0], leadPartial[:,1], s=1)
        plt.show()
        exit()

        # save leading partial dislocation
        lead = []
        for pos in xyPos:
            for j in xyPos:
                x1, y1 = pos
                x2, y2 = j
                # find the point that has the same y coordinate but different x coordinate
                #if y1 == y2 and x2-x1 != 0:
                #if y1 == y2 and np.abs(x2-x1) >= 0:
                if y1 == y2:
                    # find the point that has the bigger x coordinate
                    leadXpos = np.max([x1, x2])
                    if leadXpos == x1:
                        leadPoint = pos
                    else:
                        leadPoint = j
                    lead.append(leadPoint)
                    break
                if len(lead) == numOfPoints:
                    break
        lead = np.array(lead)
        print(lead)
        print(len(lead))
        fig, ax = plt.subplots()
        ax.scatter(lead[:,0], lead[:,1])
        plt.show()
        exit()

    return 0;

if __name__ == "__main__":
    main()
