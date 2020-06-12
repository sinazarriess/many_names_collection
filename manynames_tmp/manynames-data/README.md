# ManyNames data

## ManyNames dataset Version 1.0
###### ManyNames_v1.0 completed as of 8 January 2020

Below we describe the data that is availabe for download in this repository.

##### Notation
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

For each domain, there exists at least one WordNet category out of *article of clothing*, *instrumentality, instrumentation*, *person*, *tableware*, *ware*, *food, nutrient*, *structure, construction*, *animal*, *tool*, *food, solid food*, *plant, flora, plant life*, *vehicle*.

#### Data file: manynames-v1.0.tsv

| Column | Type | Description |
| -------- | :-------: | -------- |
| vg_image_id | int | The VG id of the image |
| vg_object_id | int | The VG id of the object |
| url | str | The url to the image, with the object marked |
| topname | str | The most frequent name in the MN responses |
| domain | str | The MN domain of the MN object |
| N | float | The number of types in the MN responses |
| perc_top | float | The relative frequency of the most frequent response (in percent) |
| H | float | The H agreement measure from (Snodgrass and Vanderwart, 1980) |
| responses | Counter | The collected MN names and their counts, i.e., the number of annotators responding them |
| singletons | dict | Contains all names which were given only once, i.e., with count=1. <br>Example: `{'sofa': 1, 'armchair': 1}` |
| vg_obj_name | str | The VG name of the object |
| vg_domain | str | The MN domain of the VG name, which may be a superset of its WN category (vg_cat, see `images.tsv`). <br>Example: The MN domain *food* subsumes the WN categories *food, solid food* and *food, nutrient*. |
| vg_synset | str | The WN synset of the object, provided by VG |

#### Data file: images.tsv
| Column | Type | Description |
| -------- | :-------: | -------- |
| vg_image_id | int | The VG id of the image |
| vg_image_name | str | The name of the VG image |
| vg_object_id | int | The VG id of the object |
| vg_obj_name | str | The VG name of the object |
| bbox_xywh | list | The coordinates of the object in the image: [left x, bottom y, width, height] <br>(y=0 is at the top of the image)|
| vg_synset | str | The WN synset of the object, provided by VG |
| vg_cat | str | The WordNet hypernym of the VG synset, corresponds roughly to one of the 7 MN domains.  |



### Citing ManyNames

Silberer, C., S. Zarrieß, G. Boleda. 2020. [Object Naming in Language and Vision: A Survey and a New Dataset](https://github.com/amore-upf/manynames/lrec2020naming.pdf). In Proceedings of LREC 2020. [[paper]](https://github.com/amore-upf/manynames/lrec2020naming.pdf)

`@inproceedings{ silberer2020manynames,`
  `title = {{Object Naming in Language and Vision: A Survey and a New Dataset}},`
  `author = {Silberer, Carina and Zarieß, Sina and Boleda, Gemma},`
  `year = {2020},`
  `booktitle = {Proceedings of the Twelfth International Conference on Language Resources and Evaluation (LREC 2020)},`
  `url = {https://www.aclweb.org/anthology/volumes/L20-1/},`
`}`

### About
ManyNames is licensed under a [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/), and based on  VisualGenome at [visualgenome.org](https://visualgenome.org).


This project has received funding from the European Research Council (ERC) under the European Union's Horizon 2020 research and innovation programme (grant agreement No 715154).
