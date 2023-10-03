# Turing_DSG_Application

### Evidence to support my application to the Turing Institute Data Study Group (December 2023)

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

The **'GPR'** folder contains three subfolders: Code, Data and Figures

* The 'Code' folder contains a demo of the MATLAB code I used in the Soft Matter paper linked below:
    * 'GPR_Predict.m' predicts the structure characteristics of a phase-separating (demixing) polymer blend from the corresponding scattering data
    * 'Pred_vs_True_Plot_Best.m' uses the output of 'GPR_Predict.m' to plot a comparison of the best predicitons of each structure characteristic with the true values
    * The remaining code belongs to the GPML MATLAB code package: http://gaussianprocess.org/gpml/code/matlab/doc/
* The 'Data' folder contains two subfolders:
    * The 'Input' folder contains the data required to run 'GPR_Predict.m'
    * The 'Output' folder contains the output of 'GPR_Predict.m' (need to run 'GPR_Predict.m' first)
* The 'Figures' folder contains the figures produced by 'Pred_vs_True_Plot_Best.m'

**Soft Matter paper**: https://pubs.rsc.org/en/content/articlelanding/2021/sm/d1sm00818h/ (Jones, Clarke; Soft Matter, 2021,
17, 9689)

*For specific details regarding our implementation of Gaussian process regression, including an overview of the data we used, please refer to sections 3.1 and 3.2 of the paper*

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

The **'Simulations'** folder contains a demo of the Julia code I used to generate the data for my PhD research:
* 'SD_Scattering_Parallel_3D' simulates spinodal decomposition in 3D, making use of parallel processing where possible, and calculates the corresponding scattering data at regular intervals
* 'Parameters' and 'Run_Simulations' are batch submission scripts used to run 'SD_Scattering_Parallel_3D' on the University of Sheffield's HPC, Stanage (https://docs.hpc.shef.ac.uk/en/latest/stanage/index.html)
    * 'Parameters' specifies the blend and simulation parameters 
    * 'Run_Simulations' specifies the number of repeat simulations, ensuring they are sent to different nodes

*For details regarding the scientific model on which the simulations are based, please refer to [Glotzer, 'Computer Simulations of Spinodal Decomposition in Polymer Blends', Annual Reviews of Computational Physics II, 1995]*

