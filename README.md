# dog_cups_analysis

Script for analyzing data for an experiment where dogs search an array of cups to find a cup containing a target odor.

Data is collected in the form of LiDAR and time to find the correct up. Conditions are wind and no wind.

This script takes all the LiDAR data and averages each cluster of points per time point by finding their centroid. A graph of the dog's trajectory as well as a heat map of most often crossed locations is produced. A basic t-test for time to find the correct cup between the no wind and wind condition is also performed.

For privacy reasons, original data is excluded from repository.
