

idx=""
outbasedir="../collected_images"$idx/
df_outfname=$outbasedir/"test_sampled_data-"$idx".csv"

python sample_images.py sample_objects $df_outfname


python sample_images.py check_images $df_outfname
echo "Extracing images ..."
. extract_images.sh $outbasedir
echo "   done"
python sample_images.py render_objects $df_outfname

outfnameBody=$outbasedir/"data_test-"$idx"_table.html"
python sample_images.py website $df_outfname $outfnameBody

for outfnameBody in $(find $outbasedir/*_table*.html); do
 outfname=$(echo $outfnameBody | sed 's/_table//g');
 cat pilot_head.html $outfnameBody pilot_tail.html > $outfname
done

