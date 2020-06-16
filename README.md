
# ManyNames data collection

This repo contains a bunch of scripts for
* selecting images useful/interesting for object naming from VisualGenome
* setting up the AMT API
* scripts for assembling the ManyNames data from AMT results
* scripts on progress on verifying the collecting data
* and some papers, which where mostly rejected :-)



**@Carina please add a short description for the folders; mark those that it would be better to delete or reorganize. **

**@all, I'm adding to-dos for each folder, with a proposal of who does what.**

Folders:

elicitation_phase0/

- initial naming data gathering via AMT, 
- including the creation of the data to be annotated and the annotation html csv and html file

evaluation/

- to evaluate an object naming model (Bottom-Up) on MN v2.0

exploration/

- to explore different datasets of referring expressions and object names

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

- elicitation_phase0/amt_phase0/ @Carina add short README.md
- analysis/ @Carina, @Sina, @Matthijs, there are to-dos for you inside the README for this folder
- dataset_creation/ @Carina, is this folder needed? If so, add short README.md?
- [DONE] verification_phase/ @Matthijs write text in existing blank README.md :) 