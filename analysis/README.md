
**results**
- Confusion matrix <br>
(see `results/domains_confusion_matrix.csv` for the full table, including collected names which were not automatically mapped to domains (marked by *))

  - rows: domains of *collected* object names
  - columns: domains of *vg* object names
  - Distribution: (normalised) counts of objects <br>
(e.g., 265 objects of the domain people (VG) were given a name of the domain clothing.)

<table border="1" class="dataframe"> 
<thead>   <tr style="text-align: right;"><th></th>     <th>max_domain</th>     <th>people</th>     <th>clothing</th>     <th>animals_plants</th>     <th>home</th>     <th>food</th>     <th>vehicles</th>     <th>buildings</th>     <th>SUM</th>   </tr> 
</thead> <tbody>   <tr>     <th>animals_plants</th>     <td>animals_plants</td>     <td>19.448</td>     <td>18.513</td>     <td>2737.833</td>     <td>58.559</td>     <td>1.397</td>     <td>8.457</td>     <td>8.079</td>     <td>2852.286</td>   </tr>   <tr>     <th>home</th>     <td>home</td>     <td>30.972</td>     <td>13.647</td>     <td>18.407</td>     <td>2700.445</td>     <td>38.393</td>     <td>5.933</td>     <td>16.371</td>     <td>2824.168</td>   </tr>   <tr>     <th>people</th>     <td>people</td>     <td>2226.020</td>     <td>80.593</td>     <td>8.440</td>     <td>37.409</td>     <td>2.930</td>     <td>6.058</td>     <td>19.067</td>     <td>2380.517</td>   </tr>   <tr>     <th>vehicles</th>     <td>vehicles</td>     <td>5.752</td>     <td>7.681</td>     <td>7.312</td>     <td>6.991</td>     <td>5.284</td>     <td>2012.692</td>     <td>27.902</td>     <td>2073.614</td>   </tr>   <tr>     <th>clothing</th>     <td>clothing</td>     <td>265.903</td>     <td>1254.441</td>     <td>18.155</td>     <td>48.122</td>     <td>0.994</td>     <td>15.389</td>     <td>23.030</td>     <td>1626.034</td>   </tr>   <tr>     <th>food</th>     <td>food</td>     <td>16.601</td>     <td>20.063</td>     <td>16.471</td>     <td>118.020</td>     <td>1230.986</td>     <td>5.732</td>     <td>4.888</td>     <td>1412.761</td>   </tr>   <tr>     <th>buildings</th>     <td>buildings</td>     <td>0.500</td>     <td>0.722</td>     <td>2.208</td>     <td>4.987</td>     <td>0.778</td>     <td>4.378</td>     <td>527.833</td>     <td>541.406</td>   </tr> </tbody></table>


**data**

- Sina's pilot:

./amt_pilot/pilot01c2019/production/40assignments_results.csv

- Collected data (pre-checkpoint) so far:

../data_phase0/

**scripts**

`python analysis.py ../data_phase0/`  -- Carina's script

`gemmas-analysis.py` -- Gemma's analysis script
