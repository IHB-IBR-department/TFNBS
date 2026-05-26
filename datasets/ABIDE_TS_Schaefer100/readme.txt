--------------
abide_ts.npy 
--------------
nested dict with structure:
	'Sub id': {'time_series': np.array(ts, roi), 
		   'label': autism or healthy_control, 
		   'site': site taken from sub id }

!! TS are not standardized !!

------------------
sch100_labels.csv
------------------
names of ROI and corresponding 7 Yeo networks
Atlas resolution - 2mm

--------------------
ABIDE preprocessing
--------------------
Loaded using nilearn. 
CPAC preprocessing:
	CompCor 5 components
	HMP 24
	GSR
	Band-pass filtering (0.01 - 0.1 Hz)
	linear and quadratic trends