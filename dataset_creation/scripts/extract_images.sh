basedirname=$1
for f in $(cat $basedirname/imgs2extract.txt); do 
    echo $f;
    if [[ $f == *VG_100K_2* ]]; then 
        unzip -p ../add_data/vgenome/images/images2.zip $f > ../add_data/vgenome/images/$f;        
    else
        unzip -p ../add_data/vgenome/images/images.zip $f > ../add_data/vgenome/images/$f;
    fi
done
