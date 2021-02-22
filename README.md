# Calculating negative emissions neccessary to offset agricultural emissions using FAiR

Analysis underlying the results presented in 

N. Brazzola, J. Wohland, A. Patt; Offsetting unabated agricultural emissions with CO2 removal to achieve ambitious climate targets, PLoS ONE, accepted manuscript

## Structure of this repository
The repository is structured as follows: 
*  `CDR_functions.py `: python file containing all the functions produced ad hoc for the analysis 
*   `main.py `: python file to execute the whole analysis with GWP100 and ERF-based offsetting for both CH4 and N2O
*   `main_metrics.py `: python file to execute the comparison of GWP100-, GWP20-, and GWP*-based offsetting for CH4 only  
*   `make_plots.py `: python file containing all the functions to produce plots 
*   `env_offset.yml `: Anaconda environment containing all libraries needed for this analysis 
* **Data**: contains all input data to the model: 
  *  *SSPs*: input data from Shared Socioeconomical Pathways 
  *  *tools*: input data necessary to calculate ensemble simulations
* **Figures**: folder where output figures are saved in 
* **Outputs**: folder where output text files are saved in 

## Steps to perform the analysis: 
1. Load the Anaconda environment as: `conda env create -f env_offset.yml`
2. In the terminal, type `ipython`
3. Run either the `main.py ` or `main_metrics.py ` file depending on which kind of analysis you want to perform 
4. Examine outputs in the folder **Figures** and **Outputs**
