import json 
import argparse
import os.path as osp
import numpy as np
import csv
import os

parser = argparse.ArgumentParser(description='Train a detector')
parser.add_argument('--files', "-f", default="_results_3D",
                     help='the dir to save logs and models')
parser.add_argument('--annot', "-a", default="3D",
                    help='the annotation type to process')
args = parser.parse_args()

loadDir = osp.join(os.getcwd(),args.files)
csvs = {}
for filename in os.listdir(loadDir):
    if filename[-4:] != '.csv':
        continue
    csv_reader = csv.reader(
            open(osp.join(loadDir,filename),'r'),
            delimiter=","
        )
    print(filename)
    csvs[filename[:-4]] = [x for x in csv_reader]

print(csvs['RoutedFusion'][0])
print(csvs['RoutedFusion'][1])


#given a prefix and suffix iterate over the items in a given list of 
# rows and produce title and values 

def getTable(csv_dict,prefix,suffix,rename=None,order=None):
    """given a prefix and suffix, generate the table of results 
    for a dictionary of csvs
    """

    titles = ['\multicolumn{1}{|c}{Easy}','\multicolumn{1}{c}{Moderate}','\multicolumn{1}{c}{Hard}']
    table = "&".join(['\multicolumn{1}{c}{Fusion Arch.}'] + titles)
    table += " \\\\ \n"


    if order == None:
       order = csv_dict.keys()
   
    for k in order:
        # print(k)
        table += k  if rename == None else rename[k]
        table += "&" + getRow(csv_dict[k],prefix,suffix)
        table += " \\\\ \n"

    return table 


def getTableTogether(csv_dict,prefixList,suffixList,rename=None,order=None):
    """given a prefix list and a suffix list it will create rows with elements of all these lists 
    """
    titles_ = ['\\textbf{Easy }','\\textbf{Moderate }','\\textbf{Hard }']
    titles_ = ['Easy','Moderate','Hard']
    titles_ = ['\multicolumn{1}{|c}{Easy}','\multicolumn{1}{c}{Moderate}','\multicolumn{1}{c}{Hard}']
    titles = []
    for x in range(len(prefixList)):
        titles += titles_

    table = "&".join(['\multicolumn{1}{c}{Fusion Arch.}'] + titles)
    table += " \\\\ \n"
   
    if order == None:
       order = csv_dict.keys()

    for k in order:
        # print(k)
        table += k if rename == None else rename[k]
        for prefix,suffix in zip(prefixList,suffixList):
            table += "&" + getRow(csv_dict[k],prefix,suffix)
        table += " \\\\ \n"

    return table 


def getRow(csv_row_list,prefix,suffix):
    indx = {x:i for i,x in enumerate(csv_row_list[0])} #by convetion 0 is header
    row = ""
    for diff in ['easy','moderate','hard']:
        try:
            i = indx[prefix+diff+suffix]
        except KeyError:
            print(prefix+diff+suffix)
            print('Error in row')
        row += meanStd(
            [float(x[i]) for x in csv_row_list[1:]]
        )
    return row[:-1]
        

def meanStd(trials,error='stderr',format_='latex'):
    N = len(trials)
    mean = np.mean(trials)
    std = np.std(trials)
    if error == 'stderr': #show standard error
        std = std/np.sqrt(N)

    if format_=='latex': #show latex formatted row
        std = "{:.2f}".format(round(std,2))
        if std[0] == "0":
            std = std[1:]

        return " ${:.2f} \\pm {}$ &".format(round(mean,2),std)
    else:
        return " {} +- {} \t".format(round(mean,2),round(std,2))



rename = {
    "replaceFusion":"Image Features Only",
    "Point-fusion-paper":"Point Fusion",
    "routed-voxel-fusion":"Routed Voxel Fusion",
    "MOE-Fusion":"MOE Fusion",
    "RoutedFusion":"Routed Point Fusion",
    "MOE-Fusion-LB":"MOE Fusion LB",
    "only-lidar":"LiDAR only baseline"
}



# prefix = "val/pts_bbox/KITTI/Car_BEV_"
# suffix = "_strict (last)"
# print(getTable(csvs,prefix,suffix,rename))


# prefix = "val/pts_bbox/KITTI/Pedestrian_BEV_"
# suffix = "_strict (last)"
# print(getTable(csvs,prefix,suffix,rename))


# prefix = "val/pts_bbox/KITTI/Cyclist_BEV_"
# suffix = "_strict (last)"
# print(getTable(csvs,prefix,suffix,rename))


# prefix = "val/pts_bbox/KITTI/Overall_BEV_"
# suffix = " (last)"
# print(getTable(csvs,prefix,suffix,rename))



# prefixList = ["val/pts_bbox/KITTI/Car_BEV_","val/pts_bbox/KITTI/Pedestrian_BEV_","val/pts_bbox/KITTI/Cyclist_BEV_"]#,"val/pts_bbox/KITTI/Overall_BEV_"]
# suffixList = ["_strict (last)","_strict (last)","_strict (last)"]#," (last)"]
# order = ['only-lidar', 'replaceFusion','Point-fusion-paper','RoutedFusion', 'routed-voxel-fusion', 'MOE-Fusion', 'MOE-Fusion-LB']
# print(getTableTogether(csvs,prefixList,suffixList,rename,order))


print("\n============== Overall Table ================")
prefix = "val/pts_bbox/KITTI/Overall_{}_".format(args.annot)
suffix = " (last)"
order = ['only-lidar', 'replaceFusion','Point-fusion-paper','RoutedFusion', 'routed-voxel-fusion', 'MOE-Fusion', 'MOE-Fusion-LB']
print(getTable(csvs,prefix,suffix,rename,order))

prefixList = ["val/pts_bbox/KITTI/Car_{}_".format(args.annot),"val/pts_bbox/KITTI/Pedestrian_{}_".format(args.annot),"val/pts_bbox/KITTI/Cyclist_{}_".format(args.annot)] #,"val/pts_bbox/KITTI/Overall_BEV_"]
suffixList = ["_strict (last)","_strict (last)","_strict (last)"]#," (last)"]
order = ['only-lidar', 'replaceFusion','Point-fusion-paper','RoutedFusion', 'routed-voxel-fusion', 'MOE-Fusion', 'MOE-Fusion-LB']
print("\n============== Car Ped Cyc Table ================")
print(getTableTogether(csvs,prefixList,suffixList,rename,order))
print(csvs.keys())