**@all please either document in this README or delete or move the files assigned to you (feel free to re-assign if needed, did the assignment as best as I could)**

analysis_verification_matthijs.py
- basic_stats()
- inter_rater_agreement() a la Krippendorff
- plot_adequacy_against_frequency()
- compute_stability_stochastic(), compute_stability_analytic(): two ways of computing and plotting the stability of the most frequent name 

filter_and_show.py
- allows you to specify rules for filtering verified names (e.g., based on adequacy and same object)
- outputs a html with a sample of the objects affected by that rule.

Sina:

compare_humans_system.ipynb
- analysis notebook for comparing object naming errors between system (bottom-up) and humans on ManyNames

wordnet_analysis.py
- a couple of utility functions for retrieving WordNet relations between pairs of names
- computing statistics for WordNet relations on naming alternatives in the original ManyNames data

wordnet_analysis_verified.py
- a couple of utility functions for retrieving WordNet relations between pairs of names
- computing statistics for WordNet relations on naming alternatives in the verified ManyNames data

[CS] @Sina, this is yours, I think:
domains_names_pairs_relations_v2.csv

Carina:

create_manynames_v1.py
create_manynames_v2.py
load_results.py  [CS] @all: this will be replaced partially by manynames.py (see manynames_tmp/), so, for loading manynames in your scripts, see manynames.py
objects_vocab-442_aliasmap.txt
?? pairs-annotation
plot_verif_data_carina.ipynb
?? rel-not-covered-verified.txt
vg_manynames.ipynb
visualise.py
agreement_table.py
aliased_MN442.txt
?? alt_objects.ipynb
analyse_verification_pilot.py

Gemma:
analysis_verification_phase.ipynb
explore.ipynb
old

**For COLING2020**

* ../plot_verif_data_carina
  - Still in need to be cleand up.


**Data**


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


