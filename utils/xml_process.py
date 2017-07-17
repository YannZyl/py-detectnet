# -*- coding: utf-8 -*-
import xml.etree.cElementTree as ET
import os
import os.path as osp 

# extract humans' properties from folder(TRAIN folder)
def extract_from_xmlfolder(folder_path):       
    data = [extract_from_singlexml(osp.join(folder_path, xmlname)) for xmlname in os.listdir(folder_path)]
    return data
    
# extract single human's properties from a xml file
def extract_from_singlexml(xml_name):
    # people's properties/directory
    human = {}
    # parse xml as a tree
    tree = ET.ElementTree(file=xml_name)
    # get image name, gender and hair length
    human['filename'] = tree.find('filename').text
    #human['gender'] = tree.find('gender').text
    #human['hairlength'] = tree.find('hairlength').text
    human['size'] = [int(tree.find('size/height').text), int(tree.find('size/width').text), int(tree.find('size/depth').text)]
    human['allobjs'] = []
    # attributions of head, top, down, shoes, hat and bag
    for elem in tree.findall('subcomponent'):
        elemname = elem.find('name').text
        if elemname != 'shoes':
            if  elem.find('bndbox/xmin').text != 'NULL':
                obj={}
                obj['name'] = elemname
                obj['bndbox'] = [int(elem.find('bndbox/xmin').text), 
                                  int(elem.find('bndbox/ymin').text), 
                                  int(elem.find('bndbox/xmax').text), 
                                  int(elem.find('bndbox/ymax').text)]
                human['allobjs'].append(obj)
        else:
            if elem.find('xmin_l').text != 'NULL':
                obj1={}
                obj1['name'] = elemname
                obj1['bndbox'] = [int(elem.find('xmin_l').text), 
                                    int(elem.find('ymin_l').text), 
                                    int(elem.find('xmax_l').text), 
                                    int(elem.find('ymax_l').text)]
                human['allobjs'].append(obj1)
            if elem.find('xmin_r').text != 'NULL':
                obj2={}
                obj2['name'] = elemname
                obj2['bndbox'] = [int(elem.find('xmin_r').text), 
                                  int(elem.find('ymin_r').text), 
                                  int(elem.find('xmax_r').text), 
                                  int(elem.find('ymax_r').text)]
                human['allobjs'].append(obj2) 
    return human

    