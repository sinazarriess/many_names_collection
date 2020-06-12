# ManyNames

Repository for the ManyNames dataset

**TODOs**:

- [ ] add file with (VG) urls of image files; add VG images
- [ ] add and describe script for WN relations@LREC paper
- [x] double-check bbox ccordinates
- [ ] add preproc scripts + raw data from computer@UPF
- [x] change incorrect column to singletons, also for MNv2.0
- [ ] column VG_cat only in MNv1.0
- [x] remove prefix from MN columns
- [ ] add short description for folders that are not yet described

### folders:
images/

- [now it's not there?]

scripts/

- see below for more information

raw_data/

- [now it's not there?]

manynames-data/

- tab-separated files with manynames data (see readme inside the folder for more information)



#### scripts/
###### Package Requirements:
  * `pandas`
  * `numpy`
  * `matplotlib.pyplot` (for `agreement_table.py` and `visualise.py`)
  * `nltk` and `nltk.corpus` (for `wordnet_analysis.py`,   `agreement_table.py`)
  * `skimage` (for `visualise.py`)

###### Usage:
All scripts can be given as optional argument the path to the ManyNames dataset: 
`python <script-name> [$MANYNAMESROOT/manynames_v1.0.tsv]`
By default, `$MANYNAMESROOT` is `../` from the script directory.
* **`manynames.py`**
  *Loads the MN data into a pandas DataFrame.*<br>
  `python manynames.py [$MANYNAMESROOT/manynames_v1.0.tsv]`
* **`visualise.py`**
  *Provides a function to draw a bounding box around an object and label it with its MN object names (and VG name).*
  You can run a demo of it with `python visualise.py`
* **`agreement_table.py`**
  *Creates a table (in tex format) of the agreement in object naming of MN. (Table 3 in the [paper](https://github.com/amore-upf/manynames/lrec2020naming.pdf)).*<br>
  `python agreement_table.py [$MANYNAMESROOT/manynames_v1.0.tsv]`
* **`plot_distr_topnames.py`**
  *Creates a stacked box plot, showing the distribution of top MN names per domain (Figure 3 in the [paper](https://github.com/amore-upf/manynames/lrec2020naming.pdf)).*<br>
  `python plot_distr_topnames.py [$MANYNAMESROOT/manynames_v1.0.tsv]`
* `wordnet_analysis.py`
  TODO
   *   **Data Prerequisites:** Download the object annotations in [objects.json.zip](https://visualgenome.org/static/data/dataset/objects.json.zip "objects.json.zip") from [VisualGenome](https://visualgenome.org "VisualGenome") (VisualGenome version 1.4) and save it under `$MANYNAMESROOT/vgenome/`.



#### raw_data/
* anonymised csvs
* TBC


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
