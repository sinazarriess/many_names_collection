
<< 'README'
    Extracts the ids
    (A) of the images of the VG images which Bottom-Up used for testing, 
        and which are covered by ManyNames. 
         The ids are saved under  
         ImageSets/mn_vg_imgids.test.txt
    (B) of the images from ManyNames for individual domains (e.g., people),  
        and saves the ids in the corresponding files at 
        ImageSets/domains/
        e.g., ImageSets/domains/people_imgids.txt
README

python scripts/create_img_splits.py

<< 'README'
Creates and saves the data that covers the joint ManyNames-VG1600 vocabulary and 
is used for evaluating an object name prediction model (Bottom-Up in this case).  
Requires as argument <mn_nameset_id>, an id for the  ManyNames target vocabulary: mnAll or mn442 (only top 442 names considered)
The files are saved at vgmn_data/<mn_nameset_id>_vg1600/
Data:
|-- vgmn_data/<mn_nameset_id>_vg1600/
    |-- <mn_nameset_id>_vg1600_vocab.tsv
    |-- <mn_nameset_id>_vg1600_MN-NMS.tsv  (MN ground truth names and info for 
                                            each object in <mn_nameset_id>-VG1600 
                                            (not necessarily required))
    |-- <mn_nameset_id>_vg1600_MNVG.tsv    (VG ground truth name and info for 
                                            each object in <mn_nameset_id> 
                                            (not necessarily required)
    |-- ImageSets       (the image ids of the images covered by VG1600, mnAll/mn442, and the test data of Bottom-Up (+ their partition into domain-specific files)
README

python scripts/prepare_eval_data.py "mnAll" # "mn442"
