

# Models to evaluate for COLING 2020
resultsdir=results/
scriptdir=scripts/
name_set="mnAll" #"mn442"
analyse=1

# FRCNN--VG1600 & VG
if [ ]; then 
    model_key="FRCNN--VG1600_VG"
    echo "Evaluating "$model_key
    outdata_path='model_output/bottom-up/resnet101_faster_rcnn_final/'
    pred_fname='scores.csv'
    imgid_fname="imgids.txt"
    #objids_dname="mn_objds.txt"
    vocab_vector_fname='objects_vocab.txt'

    if [ ! -d $resultsdir/$model_key/ ]; then
        mkdir -p $resultsdir/$model_key/;
    fi
    if [ ! -e $outdata_path/'aliased-'$vocab_vector_fname ]; then
        python $scriptdir/alias_vocabvec.py $outdata_path/$vocab_vector_fname
    fi
    
    #if [ $analyse -eq 0 ]; then 
    #    python evaluate.py --scores $outdata_path/$pred_fname --imgids $outdata_path/$imgid_fname --test 1 --verified 1 --targetvocab $outdata_path/'aliased-'$vocab_vector_fname --nameset $name_set #> $resultsdir/$model_key/'reprod_eval_verif_log-'$model_key'-'$name_set'.txt'
    #else
    python $scriptdir/eval_coling2020.py --modelkey $model_key --scores $outdata_path/$pred_fname --imgids $outdata_path/$imgid_fname --test 1 --targetvocab $outdata_path/'aliased-'$vocab_vector_fname --nameset $name_set --outfname $resultsdir/$model_key --log_file $resultsdir/'log_'$model_key #> $resultsdir/$model_key/'eval_log-'$model_key'-'$name_set'_analysis.txt';
    #fi
fi

# FRCNN--MN442 & VG
if [ "o"]; then 
    model_key="FRCNN--MN442_VG"
    echo "Evaluating "$model_key
    outdata_path="model_output/faster_rcnn_retrained/vg_442/vg_442-400-20_test/faster_rcnn_20/"
    pred_fname="mn_scores.csv"
    imgid_fname="mn_imgids.txt"
    #objids_dname="mn_objds.txt"
    vocab_vector_fname="objects_vocab-442.txt"

    if [ ! -d $resultsdir/$model_key/ ]; then
        mkdir -p $resultsdir/$model_key/;
    fi
    if [ ! -e $outdata_path/'aliased-'$vocab_vector_fname ]; then
        python $scriptdir/alias_vocabvec.py $outdata_path/$vocab_vector_fname
    fi
    
    python $scriptdir/eval_coling2020.py --modelkey $model_key --scores $outdata_path/$pred_fname --imgids $outdata_path/$imgid_fname --test 1 --targetvocab $outdata_path/'aliased-'$vocab_vector_fname --nameset $name_set --outfname $resultsdir/$model_key --log_file $resultsdir/'log_'$model_key
   # > $resultsdir/$model_key/'eval_log-'$model_key'-'$name_set'_analysis.txt';
fi

### 
# Classifiers: Fine-tuning pre-trained image features on \mn

# FRCNN--VG1600--VGMN & MN 
if [ ]; then 
    model_key="FRCNN--VG1600--VGMN_MN"
    echo "Evaluating "$model_key
    outdata_path="../../refexp_objects/output/test_classifier_vg_manynames_12epochs_vg_manynames_vis-bottomup/"
    pred_fname="scores-test_classifier_vg_manynames_12epochs_vg_manynames_vis-bottomup.csv"
    imgid_fname="mn_imgids-test_classifier_vg_manynames_12epochs_vg_manynames_vis-bottomup.txt"
    #objids_dname="mn_objds.txt"
    vocab_vector_fname="classes-test_classifier_vg_manynames_12epochs_vg_manynames_vis-bottomup.txt"

    if [ ! -d $resultsdir/$model_key/ ]; then
        mkdir -p $resultsdir/$model_key/;
    fi
    if [ ! -e $outdata_path/'aliased-'$vocab_vector_fname ]; then
        python $scriptdir/alias_vocabvec.py $outdata_path/$vocab_vector_fname
    fi
    
    python $scriptdir/eval_coling2020.py --modelkey $model_key --scores $outdata_path/$pred_fname --imgids $outdata_path/$imgid_fname --test 1 --targetvocab $outdata_path/'aliased-'$vocab_vector_fname --nameset $name_set --outfname $resultsdir/$model_key 
    > $resultsdir/$model_key/'eval_log-'$model_key'-'$name_set'_analysis.txt';
fi

# ResNet101--MN442 & MN 
if [ ]; then 
    model_key="ResNet101--MN442_MN"
    echo "Evaluating "$model_key
    outdata_path="../../refexp_objects/src/acl2020_models/vocab_mn442_resnet/"
    pred_fname="scores-test_classifier_manynames-442_4epochs_manynames-442_vis-resnet101.csv"
    imgid_fname="mn_imgids-test_classifier_manynames-442_4epochs_manynames-442_vis-resnet101.txt"
    #objids_dname="mn_objds.txt"
    vocab_vector_fname="classes-test_classifier_manynames-442_4epochs_manynames-442_vis-resnet101.txt"

    if [ ! -d $resultsdir/$model_key/ ]; then
        mkdir -p $resultsdir/$model_key/;
    fi
    if [ ! -e $outdata_path/'aliased-'$vocab_vector_fname ]; then
        python $scriptdir/alias_vocabvec.py $outdata_path/$vocab_vector_fname
    fi
    
    python $scriptdir/eval_coling2020.py --modelkey $model_key --scores $outdata_path/$pred_fname --imgids $outdata_path/$imgid_fname --test 1 --targetvocab $outdata_path/'aliased-'$vocab_vector_fname --nameset $name_set --outfname $resultsdir/$model_key 
    > $resultsdir/$model_key/'eval_log-'$model_key'-'$name_set'_analysis.txt';
fi

# ResNet101--VGMN & MN 
if [ ]; then 
    model_key="ResNet101--VGMN_MN"
    echo "Evaluating "$model_key
    base_fname="test_classifier_vg_manynames_4epochs_vg_manynames_vis-resnet101_gtmn_name"
    outdata_path="../../refexp_objects/src/acl2020_models/vocab_vgmn_resnet/"$base_fname"/"
    
    pred_fname="scores-"$base_fname".csv"
    imgid_fname="mn_imgids-"$base_fname".txt"
    #objids_dname="mn_objds.txt"
    vocab_vector_fname="classes-"$base_fname".txt"

    if [ ! -d $resultsdir/$model_key/ ]; then
        mkdir -p $resultsdir/$model_key/;
    fi
    if [ ! -e $outdata_path/'aliased-'$vocab_vector_fname ]; then
        python $scriptdir/alias_vocabvec.py $outdata_path/$vocab_vector_fname
    fi
    
    python $scriptdir/eval_coling2020.py --modelkey $model_key --scores $outdata_path/$pred_fname --imgids $outdata_path/$imgid_fname --test 1 --targetvocab $outdata_path/'aliased-'$vocab_vector_fname --nameset $name_set --outfname $resultsdir/$model_key 
    > $resultsdir/$model_key/'eval_log-'$model_key'-'$name_set'_analysis.txt';
fi

# ResNet101--VGMN & VG 
if [ ]; then 
    model_key="ResNet101--VGMN_VG"
    echo "Evaluating "$model_key
    outdata_path="../../refexp_objects/src/acl2020_models/vocab_vgmn_resnet/gt_vgname/"
    base_fname="test_classifier_vg_manynames_4epochs_vg_manynames_vis-resnet101_gtvg_name"
    #base_fname="test_classifier_vg_manynames_8epochs_vg_manynames_vis-resnet101_gtvg_name_cont"
    
    pred_fname="scores-"$base_fname".csv"
    imgid_fname="mn_imgids-"$base_fname".txt"
    #objids_dname="mn_objds.txt"
    vocab_vector_fname="classes-"$base_fname".txt"

    if [ ! -d $resultsdir/$model_key/ ]; then
        mkdir -p $resultsdir/$model_key/;
    fi
    if [ ! -e $outdata_path/'aliased-'$vocab_vector_fname ]; then
        python $scriptdir/alias_vocabvec.py $outdata_path/$vocab_vector_fname
    fi
    
    python $scriptdir/eval_coling2020.py --modelkey $model_key --scores $outdata_path/$pred_fname --imgids $outdata_path/$imgid_fname --test 1 --targetvocab $outdata_path/'aliased-'$vocab_vector_fname --nameset $name_set --outfname $resultsdir/$model_key 
    > $resultsdir/$model_key/'eval_log-'$model_key'-'$name_set'_analysis.txt';
fi



# ResNet101--MN442 & VG 
if [ ]; then 
    model_key="ResNet101--MN442_VG"
    echo "Evaluating "$model_key
    outdata_path="../../refexp_objects/src/acl2020_models/vocab_mn442_resnet/gt_vgname/"
    pred_fname="scores-test_classifier_manynames-442_8epochs_manynames-442_vis-resnet101_gtvg_name_cont.csv"
    imgid_fname="mn_imgids-test_classifier_manynames-442_8epochs_manynames-442_vis-resnet101_gtvg_name_cont.txt"
    #objids_dname="mn_objds.txt"
    vocab_vector_fname="classes-test_classifier_manynames-442_8epochs_manynames-442_vis-resnet101_gtvg_name_cont.txt"

    if [ ! -d $resultsdir/$model_key/ ]; then
        mkdir -p $resultsdir/$model_key/;
    fi
    if [ ! -e $outdata_path/'aliased-'$vocab_vector_fname ]; then
        python $scriptdir/alias_vocabvec.py $outdata_path/$vocab_vector_fname
    fi
    
    python $scriptdir/eval_coling2020.py --modelkey $model_key --scores $outdata_path/$pred_fname --imgids $outdata_path/$imgid_fname --test 1 --targetvocab $outdata_path/'aliased-'$vocab_vector_fname --nameset $name_set --outfname $resultsdir/$model_key 
    > $resultsdir/$model_key/'eval_log-'$model_key'-'$name_set'_analysis.txt';
fi



