**@Sina, @Matthijs, @Gemma: adapt paths of remaining scripts to the new folder structure of the repo**

### Folders:

#### analysis/
A set of scripts and jupyter notebooks for analysing ManyNames (v1.0 und v2.0) and its verification annotations. 

##### Files:

[TODO CS] Carina:

[TODO CS] plot_verif_data_carina.ipynb

[TODO CS] vg_manynames.ipynb

[TODO CS] agreement_table.py
- Computes the agreement table that is given in Silberer et al., (LREC, 2020).

[TODO CS] analyse_verification_pilot.py


analysis_verification_matthijs.py

- basic_stats()
- inter_rater_agreement() a la Krippendorff
- plot_adequacy_against_frequency()
- compute_stability_stochastic(), compute_stability_analytic(): two ways of computing and plotting the stability of the most frequent name 

filter_and_show.py
- allows you to specify rules for filtering verified names (e.g., based on adequacy and same object)
- outputs a html with a sample of the objects affected by that rule.

compare_humans_system.ipynb
- analysis notebook for comparing object naming errors between system (bottom-up) and humans on ManyNames

gemmas_analysis_verified_data.ipynb

- contains some stats and plots of the verified data, pre ManyNames v. 2, most notably distribution of adequacy, distribution of raw number of adequate names, distribution of the frequency of the entry-level name in the canonical object. 

wordnet_analysis.py

- a couple of utility functions for retrieving WordNet relations between pairs of names
- computing statistics for WordNet relations on naming alternatives in the original ManyNames data

wordnet_analysis_verified.py
- a couple of utility functions for retrieving WordNet relations between pairs of names
- computing statistics for WordNet relations on naming alternatives in the verified ManyNames data

alt_objects.ipynb
- notebook for exploring the bounding box issue with VisualGenome: the same object can have many different boxes
- functions for plotting boxes, computing iou, finding candidates for the same-obj-diff-box pattern 


#### visual-relationships-errors/

- manual exploration of the relationship between alternative objects being named in ManyNames v.1. Deprecated -- here for legacy purposes and in case someone wants to retake it at some point.

### Files:

[TODO CS] load_results.py  / manynames.py
  [CS] @all: this will be replaced partially by manynames.py (see manynames_tmp/), so, for loading manynames in your scripts, see manynames.py

- Loads ManyNames v1.0 or v2.0, or the cleaned MN, containing the cerification annotations etc.  


create_manynames_v1.py

create_manynames_v2.py

visualise.py


**For COLING2020**

* ../plot_verif_data_carina
[TODO CS]  - Still in need to be cleaned up.


**Data**

- objects_vocab-442_aliasmap.txt
    - Mapping of the 442 MN entry-level names to the aliased object names of VisualGenome, which were used for Bottom-Up (Anderson et al., 20XX).

- aliased_MN442.txt
    - The list of VG-aliased 442 MN entry-level names (see objects_vocab-442_aliasmap.txt)

* see `raw_data_phase0` for raw data collected from AMT
* see `proc_data_phase0`for preprocessed data
  - `proc_data_phase0/basic_preprocessing/all_results_files_preprocessed_rounds0-3.csv` is essentially the basic ManyNames dataset
* some of the scripts in this folder might have outdated paths to data files
* proc_data_phase0/verification/all_responses_round0-3_verified.csv adds verification data.

The 'verified' column contains, for each image (row), a dictionary from each name for that image to:
-   cluster (list of names),
-   adequacy (mean, normalized, with 0=inadequate and 1=adequate),
-   inadequacy_type (majority vote; None if name is adequate),
-   cluster_id (0 = the most prominent cluster, based on all names in each cluster; 1 the next prominent cluster, etc.),
-   cluster_weight (so the proportion of people who entered a (any) name in that cluster, proportion between 0 and 1)

Example: {'man': {'cluster': ('baseball player', 'batter', 'man', 'person', 'player'), 'adequacy': 1.0, 'inadequacy_type': None, 'cluster_id': 0, 'cluster_weight': 0.9375}, 'helmet': {'cluster': ('helmet',), 'adequacy': 0.5, 'inadequacy_type': 'bounding box', 'cluster_id': 1, 'cluster_weight': 0.0625}, 'player': {'cluster': ('baseball player', 'batter', 'man', 'person', 'player'), 'adequacy': 1.0, 'inadequacy_type': None, 'cluster_id': 0, 'cluster_weight': 0.9375}, 'batter': {'cluster': ('baseball player', 'batter', 'man', 'person', 'player'), 'adequacy': 1.0, 'inadequacy_type': None, 'cluster_id': 0, 'cluster_weight': 0.9375}, 'baseball player': {'cluster': ('baseball player', 'batter', 'man', 'person', 'player'), 'adequacy': 1.0, 'inadequacy_type': None, 'cluster_id': 0, 'cluster_weight': 0.9375}, 'person': {'cluster': ('baseball player', 'batter', 'man', 'person', 'player'), 'adequacy': 1.0, 'inadequacy_type': None, 'cluster_id': 0, 'cluster_weight': 0.9375}}


