


** Step 1:

preprocess all result csvs and create a single csv file, augmented with our domain labels:<br>
`basic_preprocessing/preprocess_results.py all_results_files_preprocessed_roundX.csv` <br>


** Step 2:

run the spellchecker
`spellchecking/spellcheck-dataframe.py`

** Pruning

this was done during phase0 for excluding images that where problematic due to occlusions, etc. right?
