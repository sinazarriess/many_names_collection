
phase="-phase0"
idx="-sampling2"$phase

outbasedir="../collected_images"$idx/
df_outfname=$outbasedir/"sampled_data-html.csv"
#"pilot_sampled_data-amt_wtaboolist.csv"

python -i sample_images.py sample_objects $df_outfname
# create df from objIDs file
#python sample_images.py create_df $df_outfname
return
python sample_images.py check_images $df_outfname
echo "Extracing images ..."
. extract_images.sh $outbasedir
echo "   done"
python sample_images.py render_objects $df_outfname

outfnameBody=$outbasedir/"data_test"$idx"_table.html"

return
python sample_images.py website $df_outfname $outfnameBody

for outfnameBody in $(find $outbasedir/*_table*.html); do
 outfname=$(echo $outfnameBody | sed 's/_table//g');
 cat 'pilot_head'$phase'.html' $outfnameBody 'pilot_tail.html' > $outfname
done

