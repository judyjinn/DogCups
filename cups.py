"""
@author: judyjinn

Dog Cup Experiment  Data

Takes LiDAR data from dogs searching in a 55 cup array for target cup,
Graphs a heat map or trajectory map of LiDAR data after finding centroids for
each time point.

Also calculates quick stats of wind vs no wind conditions in experiment.

"""

'''                         #######################
#------------------------   ## ---   Set up  --- ##     ------------------------
                            #######################
'''
"""import every package you'll need"""
import os 
import re #used for natural sort function

import pandas as pd  #data frames

import matplotlib
matplotlib.use("TkAgg") # Need to do this for tkinter to properly import and
                        # allow simulataneous matplotlib figures
from matplotlib import pyplot as plt
import matplotlib.animation as animation #animate plots, unused
import matplotlib.cm as cm
import matplotlib.patches as patches

import tkinter as Tk #opening files
from tkinter import filedialog

from ggplot import *

import numpy as np

import scipy.stats as stats

pd.set_option('display.max_columns',0) # auto-detect width of terminal 
plt.ion() # Solves that iPython matplotlib hang problem
pd.options.mode.chained_assignment = None # turns off the chaining warning 

'''                          #######################
#-------------------------   ## ---   Notes   --- ##     -----------------------
                             #######################
'''

'''

TODO:   Currently heat maps and just graph maps are using all trials in one data 
        frame, thus an extra line is being drawn across from the end of one trial 
        to the beginning of the next. Make sure to graph future lines as 
        seperate plots.

'''



'''                           #######################
#--------------------------   ## --- Functions --- ##     ----------------------
                              #######################
'''

def open_folder():
    ''' Open folder using GUI

    Args:
        None

    Returns: 
        file_path:    str; path to folder
    '''
    
    root = Tk.Tk()
    #stop extra root window from Tk opening
    root.withdraw()
    #close Tk file open  after selecting file and allow matplotlib to plot.
    root.update()
    file_path = Tk.filedialog.askdirectory()
    root.quit() 
    
    return file_path

def natural_sort(l): 
    ''' Sorts data alphabetically
    '''
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)
    
def condition(data):
    ''' Gets trial condition, wind or no wind

    Args:
        data:         pandas data frame

    Returns: 
        condition:    str; condition type
    '''
    
    if data in [6,7,10,12,13]:
        condition = 'wind'
    else:
        condition = 'no wind'
    return condition    
    
def get_sec(x):
    ''' Transforms values from minutes, seconds and milliseconds into seconds

    Args:
        x:      str; form mm:ss:ff

    Returns: 
        sec:    int; time in seconds
    '''
    
    if pd.isnull(x):
        sec = np.nan
    else:
        m, s, ff = x.split(':')
        sec= (int(m)*60) + (int(s) + (float(ff)/100))
    return sec

def get_csv_names(folder):
    ''' Read all the names of files in a folder and store as list

    Args:
        folder:      str; path to folder of interest

    Returns: 
        csv_names:  list of str; contains file names
    '''

    csv_names_temp = os.listdir(folder)
    csv_names = os.listdir(folder)

    # Remove hidden folders and also any files that are not necessary (lack _)
    for f in csv_names_temp:
       if (f.startswith('.')==True):
           csv_names.remove(f)
       elif '_' not in f:
           csv_names.remove(f)
    return csv_names
    

def org_data(csv_folder):
    ''' Clean data by finding centroids of each cluster of points per time frame

    Args:
        csv_folder: str; path to folder of interest

    Returns: 
        all_ldr_dict:   dict; a dictionary of pandas data frames
                        contains all original LiDAR data from all dogs
                        each key is one dog trial
        centroid_dict:  dict; a dictionary of pandas dat aframes
                        contains centroid averaged LiDAR data from all dogs
        
    '''
    
    csv_names = get_csv_names(csv_folder)
    
    all_ldr_dict = {}
    centroid_dict = {}
    
    # Loop through all dog LiDAR files
    for fname in csv_names:  
        name = fname.split("_")[0]
        trial = fname.split("_")[1].split(".")[0]        
        
        ldr_data = pd.read_csv(csv_folder+"/"+fname, sep=",")
        
        # Add dog name and trial number to pandas data frame
        ldr_data['dog'] = name
        ldr_data['trial'] = int(trial)
    
        # # change all -1 to NaN then forward fill gap with values
        ldr_data['TIMESTAMP'] = ldr_data['TIMESTAMP'].replace('-1', np.nan)
        ldr_data=ldr_data.ffill()
        
        # Find x and y euclidean distances of each point from the radians
        ldr_data['rad'] = np.deg2rad(ldr_data['AZIMUTH'])
        ldr_data['X'] = (ldr_data['DISTANCE']*np.cos(ldr_data['rad']))
        ldr_data['Y'] = (ldr_data['DISTANCE']*np.sin(ldr_data['rad']))

        # Remove walls of room
        ldr_data = ldr_data[(ldr_data['Y']>43.0) & (ldr_data['Y']<1100.0)]
        ldr_data = ldr_data[(ldr_data['X']>-200.0) & (ldr_data['X']<280.0)]

        # Group all points from same time point
        t_grp = ldr_data[['X', 'Y']].groupby(ldr_data['TIMESTAMP'])
        # Next line only used to print if you want to see the groups
        t_grp_list = list(ldr_data[['X','Y']].groupby(ldr_data['TIMESTAMP']))
        # get number of observed points in each time group
        t_grp_size =ldr_data[['X', 'Y']].groupby(
            ldr_data['TIMESTAMP']).size().reset_index() 
        t_grp_size.columns=['TIMESTAMP', 'XYgrp_size']
        
        # Find total "mass" of the x/y points by time point
        t_grpX = ldr_data['X'].groupby(ldr_data['TIMESTAMP']).sum()
        t_grpY = ldr_data['Y'].groupby(ldr_data['TIMESTAMP']).sum()

        centroid_XY = pd.concat([t_grpX, t_grpY], axis=1).reset_index()
        # merge length of each group back to data frame
        centroid_XY= pd.merge(centroid_XY, t_grp_size, on='TIMESTAMP') 
        
        # Find the centroid of each cluster of points. 
        # Add 170 to the X and 50 to Y to place the graph with the origin 
        # for the lower left corner
        centroid_XY['center_x'] = (centroid_XY['X']/centroid_XY['XYgrp_size'])+170
        centroid_XY['center_y'] = (centroid_XY['Y']/centroid_XY['XYgrp_size'])-50
        
        # Label new data frame with dog name, trial number, and the condition
        centroid_XY['dog'] = name
        centroid_XY['trial'] = int(trial)
        centroid_XY['condition']=centroid_XY['trial'].apply(condition)
        
        dict_key = fname.split(".")[0]
        all_ldr_dict[dict_key]=ldr_data
        centroid_dict[dict_key]=centroid_XY
    
    return all_ldr_dict, centroid_dict


def concat_all(folder):
    ''' Concatenates all data into single data frame

    Args:
        folder: str; path to folder of interest

    Returns: 
        all_ldr_df:     data frame; data frame of all original LiDAR data
        centroid_df:    data frame; data frame of centroid LiDAR data
        
    '''
    
    # Open all data and store LiDAR data as dictionary
    all_ldr_dict, centroid_dict = org_data(folder)
    
    all_names = list(all_ldr_dict.keys())
    
    temp_all_ldr_df = all_ldr_dict[all_names[0]]
    temp_centroid_df = centroid_dict[all_names[0]]

    # concat into data frames
    for name in all_names[1:]:
        temp_all_ldr_df = pd.concat([temp_all_ldr_df, all_ldr_dict[name]])
        temp_centroid_df = pd.concat([temp_centroid_df, centroid_dict[name]])

    # Save as CSVs    
    all_ldr_df = temp_all_ldr_df
    all_ldr_df.to_csv("all_ldr.csv"  , sep=',',index=False)
    
    centroid_df = temp_centroid_df
    centroid_df.to_csv("centroid_df.csv"  , sep=',',index=False)
      
    return all_ldr_df, centroid_df
    
    
def correct_cup(trial):
    ''' Finds position of correct cup for each trial

    Args:
        trial:  int; trial number

    Returns: 
        cup:    list of ints; X-Y coordinates of cup location    
    '''
    
    if (trial==6) | (trial==14):
        cup = [100,300]
    elif (trial==7) | (trial==11):
        cup = [300,0]
    elif (trial==8) | (trial==13):
        cup = [400,200]
    elif (trial==9) | (trial==12):
        cup = [200,100]
    elif (trial==10) | (trial==15):
        cup = [300,400]
    return cup

def just_graph(data, grid_array, trial):
    ''' Graph the movement data for each trial and saves graph

    Args:
        data: pandas data frame; all data for all trials
        grid_array: numpy array; contains coordinates of all 55 cup locations
        trial: int; trial number of interest

    Returns: 
        None   
    '''

    data=data[data['trial']==trial]
    cup = correct_cup(trial)

    # Get data for where participant moved during trial and angle of view
    loc_x = data['center_x']
    loc_y = data['center_y']

    # Specific color palette
    cmap = cm.jet

    # Set up figure
    # Create figure
    fig = plt.figure()
    ax = fig.gca()

    ax.set_xlim(-50,500)
    ax.set_ylim(-50, 1100)
    plt.style.use('ggplot')
    ax.set_axis_bgcolor('white')
    ax.set_aspect('equal')
    # ax.grid(color='lightgray', linestyle='-', linewidth=1)
    
    
    plt.scatter(grid_array[:,0], grid_array[:,1], color='gray', s=10, zorder=0)
    plt.scatter(cup[0], cup[1], color='red', s=15, zorder=2)
    
    # Plot just gray lines 
    plt.plot(loc_x, loc_y, linestyle='-', c='lightgray',  zorder=1)
    
    
    # # Plot colored dots by progress
    # step = 1.0/float(len(loc_x))
    # for i in range(0,len(loc_x)):
    #
    #     # Gives the color of the plot point. basically it's % of trajectory complete * 255 = color on 0-255 scale
    #     c = cmap(int(np.rint(i*step * 255)))
    #     plt.scatter(loc_x[i], loc_y[i], color=c, edgecolors='none', zorder=1)

    # Save it to the directory created earlier
    # plt.savefig(graph_folder+"/"+codename+"_"+trial, transparent=True)

    fig.tight_layout()
    path=os.getcwd()+'/graphs/'
    fig.savefig(path+'trajectory_'+str(trial)+'.png')
    plt.show()

    return


    
def heat_map(data, grid_array, trial):
    ''' Graphs heat map of most traversed locations for a trial and saves graph

    Args:
        data: pandas data frame; all data for all trials
        grid_array: numpy array; contains coordinates of all 55 cup locations
        trial: int; trial number of interest

    Returns: 
        None   
    '''
    
    cup = correct_cup(trial)
    data=data[data['trial']==trial]
    condition=data['condition'].iloc[0]

    # Get data for where participant moved during trial and angle of view
    loc_x_all = data['center_x']
    loc_y_all = data['center_y']
    
    heatmap, xedges, yedges = np.histogram2d(loc_x_all, loc_y_all, bins=40)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    
    fig = plt.figure()
    ax = fig.gca()
    ax.set_xlim(-50, 500)
    ax.set_ylim(-50, 1100)
    plt.style.use('ggplot')
    #set background to the same min color val of the heatmap
    ax.set_axis_bgcolor(color = '#ffffcb') 
    ax.set_aspect('equal')
    ax.grid(visible=False)
    
    plt.title('Trial '+str(trial)+' with '+condition)

    im = plt.imshow(heatmap.T, extent=extent, cmap='YlOrRd', origin='lower')
    plt.scatter(grid_array[:,0], grid_array[:,1], color='gray', s=10, zorder=1)
    plt.scatter(cup[0], cup[1], color='red', s=25, zorder=3)
    
    dog_names = list(centroid['dog'].unique())
    for name in dog_names:
        dog = data[data['dog']==name]
        loc_x = dog['center_x']
        loc_y = dog['center_y']
        plt.plot(loc_x, loc_y, linestyle='-', c='black',  zorder=2, alpha=0.5)
    
    path=os.getcwd()+'/graphs/'
    fig.savefig(path+'heatmap_'+str(trial)+'.png')
    plt.show()


    return



def org_cups(cup_df):
    ''' Cleans data for data frame containing times for each trial.

    Args:
        cup_df:     pandas data frame; times for each dog and trial

    Returns: 
        cup_data:   pands data frame; organized data with names
    '''
    
    # Transform data into pandas-friendly format
    cup_data = pd.melt(cup_df, id_vars='Dog').sort_values(by='Dog')
    
    cup_data.columns = ['dog', 'trial', 'time']
    # Get trial number from string
    cup_data['trial'] = cup_data['trial'].apply(lambda x: int(x[4:]))
    # Label with correct wind condition
    cup_data['condition'] = cup_data['trial'].apply(condition)
    # Transform time from str to seconds
    cup_data['time'] = cup_data['time'].apply(lambda x: get_sec(x))
    
    return cup_data

'''                                         #######################
#----------------------------------------   ## ---    MAIN   --- ##     -----------------------------------------
                                            #######################
'''

if __name__ == '__main__':

    # Concatenate all data if running for first time
    # csv_folder = open_folder()
    # all_ldr_dict, centroid_dict = org_data(csv_folder)
    # all_ldr, centroid = concat_all(csv_folder)

    # Open single file for testing
    centroid =pd.read_csv(
        '/Users/judyjinn/Python/DogTracking/Cups/centroid_df.csv', sep=','
        )
    
    # Mark grid, offset (-170,50) to account from to LiDAR
    rows = np.arange(0,11)*100 #+50
    rows_repeat = np.repeat(rows,5)
    columns = np.arange(0,5)*100 #-170
    columns_tile = np.tile(columns,11)
    
    grid_array = np.column_stack((columns_tile,rows_repeat))

    # Graph heat maps for trials 6-16 for all dogs
    for i in range(6,16):
        # just_graph(centroid,grid_array,i)
        heat_map(centroid, grid_array,i)

    
    # Quick cup data stats
    cup_df =pd.read_csv(
        '/Users/judyjinn/Python/DogTracking/Cups/cups_data.csv', sep=','
        )
    cup_data = org_cups(cup_df)

    no_wind = cup_data[
        cup_data['condition']=='no wind'
        ].groupby('dog').mean().reset_index()
    wind = cup_data[
        cup_data['condition']=='wind'
        ].groupby('dog').mean().reset_index()
    t, p =stats.ttest_ind(no_wind['time'], wind['time'], equal_var=True)
    print('T-test for Time (sec) to Find Cup in No Wind vs Wind Conditions')
    print('No Wind Conditon: ',no_wind['time'].mean(),'±', 
        no_wind['time'].std(), '   n =', len(no_wind['dog'].unique())
        )
    print('Wind Conditon: ',wind['time'].mean(),'±', wind['time'].std(), 
        '   n =', len(wind['dog'].unique())
        )
    print('T =',t,'   p =',p)


    
    
    