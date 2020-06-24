
# ManyNames data collection

This repo contains a bunch of scripts for
* selecting images useful/interesting for object naming from VisualGenome
* setting up the AMT API
* scripts for assembling the ManyNames data from AMT results
* scripts on progress on verifying the collecting data
* and some papers, which where mostly rejected :-)

### Folders:

elicitation_phase0/

- initial naming data gathering via AMT, 
- including the creation of the data to be annotated and the annotation html csv and html file

evaluation/

- to evaluate an object naming model (Bottom-Up) on MN v2.0

exploration/

- to explore different datasets of referring expressions and object names (what we did in the beginning when we were deciding where to get images, names from)

manynames_tmp/

- folder to prepare the public data and script release(s) of ManyNames

papers/

- contains, well, aeh..., papers. :) 

proc_data_phase0/

- contains processed data and scripts to process it. Goes from raw_data to ManyNames (**@Carina, correct me if I'm wrong**). Includes scripts to add verification column. File `all_results_files_preprocessed_rounds0-3.csv' is essentially the basic ManyNames dataset.

scripts/

- to load and analyse ManyNames v1.0, v2.0, and the cleaned, crowd-sourced and data (excluding or including the verification annotations)

verification_phase0/	scripts and data from the verifcation phase. 

Files:

[TODO CS] plot_verif_data_carina.ipynb  **@Carina, can we remove this? If not, move somewhere else?**

**TO-DOs:**

- elicitation_phase0/amt_phase0/ @Carina verifies that all files are the latest version and adds short README.md
- elicitation_phase0/dataset_creation/  @Carina verifies that all files are the latest version and adds short README.md
- evaluation/ @Carina adds data files and fixes minor bug
- proc_data_phase0/ @Carina checks that all files are latest version, possibly adapts paths
- scripts/ see readme there for @Carina's to-dos.
- @Carina checks to-dos for manynames_tmp (see README.md there)
- **@Sina, @Matthijs, @Gemma: adapt paths of remaining scripts to the new folder structure of the repo**