This is a regression model for three different models of nano-particle flow. The math that corresponds to this can be found in math.pdf. 

There is both a python and matlab version that should produce similar results. The python version should be run as follows: 
python biexponential.py ../data/synthetic.csv
where synthetic.csv corresponds to the data file to fit it on, and biexponential.py can be replaced with the model that you want to fit.
For Matlab I hardcoded in the datafile name because Matlab does not play nicely with command line arguments. So change the datafile name on the second line of the matlab file to the one that you want.

The output of each is the parameters and the sum of squared errors (I know the python variabel name is mse, but it is actually sse).
