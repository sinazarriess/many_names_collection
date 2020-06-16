#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 15:57:05 2019

@author: Carina Silberer
"""

from collections import Counter

img_web_dir = "http://object-naming-amore.upf.edu/"

def _fill_html_table_row_analysis(obj_name, domain, collected_names, top_domain, meta, image_url, item_id):
    col_img_src = '<img src={0} alt="{1}" style="width:350px;height:350px;">'.format(image_url, image_url.split("/")[-1])
    col_coll_names = '<font color="black"><b>Coll. Names/Words</b><br>{0}</font><br><br>\n\
    <b>Top Domain:</b><br>{1}<br>\n'.format(collected_names, top_domain)
        
    col_obj_name = '<b>VG Object Name</b>: <br>{0}<br><br>\
    <b>VG Domain</b>: <br>{1}<br><BR>'.format(obj_name, domain)
    
    col_meta = '<b>Meta Info</b>: <br>{0}<br>'.format(meta)
        		
    html_row = '<tr>\n\t\
    <td width="50">%s</td>\n\t\
    <td align="position" valign="position" width="150"><div style="background-color:#ffff99">%s</div></td>\n\t\
    <td width="300">%s</td>\n\t\
    <td width="300">%s</td>\n\t</tr>\n\t\
    ' % (col_img_src, col_obj_name, col_coll_names, col_meta)
    return html_row

def write_html_table(data_df, html_fname):
    """
    Columns of data_df:
        ['vg_img_id', 'vg_object_id', 'vg_obj_name', 'vg_domain',
       'top_response_domain', 'responses', 'url', '#bbox', '#other',
       '#other_reasons', '#occl', 'plurals', 'singulars', 'domain_match']
    """
    table = '<table style="width:100%" cellpadding="10" cellspacing="10" frame="box">\n<tbody>'
    
    row_num = 0
    for row in data_df.iterrows():
        image_url = row[1]["url"]
        obj_name = row[1]["vg_obj_name"]
        domain = row[1]["vg_domain"]
        criteria = "pl: {0.plurals}<br>\n\
            occl: {0[#occl]}<br>\n\
                    bbox: {0[#bbox]}<br>\nother: {0[#other]}<br>".format(row[1])
        if "responses" in data_df.columns: 
            coll_obj_names = _pretty_obj_names(row[1]["responses"])
            top_domain = row[1]["top_response_domain"]
        else:
            coll_obj_names = "--"
            top_domain = "--"
            
        html_row = _fill_html_table_row_analysis(obj_name, domain,
                                                 coll_obj_names, top_domain,
                                                 criteria,
                                                 image_url, row_num)
        table += '\t%s\n' % html_row 
        row_num += 1
        
    table += "</tbody></table>\n"

    with open(html_fname, "w") as f:
        f.write(table)
        
def _pretty_obj_names(obj_names):
    if isinstance(eval(obj_names), Counter):
        obj_names = "/".join(["%s: %d" % (name, count) for (name, count) in eval(obj_names).items()])
    elif isinstance(eval(obj_names), dict):
        obj_names = "/".join(["%s: %s" % (name, str(counts)) for (name, counts) in eval(obj_names).items()])
    elif isinstance(eval(obj_names), list):
        obj_names = "/".join([str(l) for l in eval(obj_names)])
    return obj_names.replace("/", "<br>")