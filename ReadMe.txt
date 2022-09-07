Requirements and dependencies:
    1.Numpy
    2.matplotlib.pyplot
    3.from matplotlib.patches import Circle
    4.glob
    5.from skimage.transform import resize

Loading train and test data:

I have copied the Images,Ground Truths folder datasets from UCI-JARTOOL dataset files into current working directory.
Please place Images,Ground Truths folders in the current working directory
I use custom functions (Load_Image, Load_lxyr, crop_images) to load and crop volcanic image pathces and convert to numpy arrays


Running instructions:

run the following commands:
	1)for data exploration	:	Plots.py
	2)for experiments		:	main.py

All supporting functions are mainly located in utilities.py


Modification instructions:

1)To modify crop size					:	utilities.py-->crop_images()
2)To modify number of principle components 	:  	main.py--->PCA callable function(in all experiments)
