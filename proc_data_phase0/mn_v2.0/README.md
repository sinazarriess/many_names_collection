
# ManyNames

## Version 2.0 of dataset (completed as of 20 April 2020)

### Notation
**MN** Abbreviation for ManyNames  
**VG** Abbreviation for VisualGenome  
**domain** The MN categorisation of objects into  
   * people
   * animals_plants
   * vehicles
   * food
   * home
   * buildings
   * clothing
   
**verif-annos** Abbreviation for the annotations collected in the verification crowd-sourcing, includes  
   * adequacy of the name, given a target object
   * if inadequate, what could be the possible source of error  
     (bounding box, visual, linguistic, other)  
     if adequate, the inadequacy_type is "nan"  
   * same-object: set of names referring to the same object

### manynames-v2.0_public.csv

| Column | Type | Description | 
| -------- | :-------: | -------- |
| vg_image_id | int | The VG id of the image |
| vg_object_id | int | The VG id of the object |
| url | str | The url to the image, with the object marked | 
| mn_topname | str | The most frequent name in the MN responses |
| mn_domain | str | The MN domain of the MN object |
| responses | Counter | The collected, correct MN names and their counts, i.e., the number of annotators responding them | 
| same_object | dict (dict(str:float)) | For each correct MN name n, the correct MN names referring to the same object as n, and their normalised counts of the same-obj votes |
| adequacy_mean | dict (str:float) | The correct MN names and their mean adequacy score |
| inadequacy_type | dict (str:float) | The distribution of the inadequacy types of all correct MN names (mean score across verif-annos) |
| incorrect | dict (dict) | Complex dict, see the notes below. | 
| vg_obj_name | str | The VG name of the object |
| vg_domain | str | The MN domain of the VG name |
| vg_synset | str | The VG synset of the object |
| vg_cat | str | The WordNet hypernym of the VG synset |
| vg_same_object | dict (str:float) | The MN names which refer to the same object that the VG name identifies. and the normalised counts of the samoe-obj-votes |
| vg_adequacy_mean | float | The mean adequacy score of the VG name |
| vg_inadequacy_type | dict (str:float)| The distribution of the inadequacy types of the VG name (mean score across verif-annos) |


#### Notes
   * A name is considered correct if its mean adequacy score > 0.4
   * Names are considered to refer to the same object if their pairwise mean same-object score > 0
   * The `incorrect` column contains all incorrect names, together with the MN annotations: their count, mean adequacy, inadequacy_types, same_object information
