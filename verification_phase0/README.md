0_internal_annotation/
- html and csv files, with py script creating them, for an internal annotation round we did prior to crowdsourcing. 

1_crowdsourced/
- html, ini, csv files for the crowdsource task
- raw results of the crowdsource task (only anonymized; with names only on external harddisk)

amt_api/
- scripts for interacting with the amazon mechanical turk api

qualification/
- .xml files representing a mechanical turk 'qualification' task, which workers had to do prior to the real thing. 

test_imgids/
- Lists of image ids to collect names first for. See also readme there. These lists are used by create_amt_csv_for_verification.py.

unique_worker_hits/
- files for creating a single hit for a worker (e.g., for extra payment)

align_with_manual_annotations.py
- To test how the crowdsource results and our internal annotations (mis)align.

create_amt_csv_for_verification.py
- Creates a .csv with items to be turned into mechanical turk HITs; these are found in 1_crowdsourced/ .

merge_and_clean_csv.py
- Combines results csv files (as created by results_to_csv.py) from all batches of HITS.
- Filters out unreliable assignments.
- Anonymizes worker names.
- Writes a new csv with the combined, cleaned results.


results_to_csv.py
- Turns mturk results into a .csv file;
- Prints semi-usable feedback for individual workers in case they request it. 