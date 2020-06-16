


1. Prepare evaluation data used for evaluation
   --> prepare_eval_data.sh
   
   Required data:
    * vg_data/objects_vocab.txt
    * model_output/bottom-up/resnet101_faster_rcnn_final/test.txt
      OR
      ImageSets/mnAll_vg1600_imgids.test.txt
      
2. Run evaluation
   --> evaluate_coling2020.sh
   
   Scripts used:
   * eval_coling2020.py (main script)
   * utils_vocab.py
   * mn_loader.py
   * alias_vocab.py (need required for Bottom-Up, but if using another model trained with different vocab)

