import json
import csv
from pathlib import Path
import numpy as np
import cv2

def angle_axis_to_opk(angle_exis):
    # ODM orientations are in angle/axis format (see https://github.com/mapillary/OpenSfM/issues/121)
    R = cv2.Rodrigues(np.array(angle_exis))[0].T
    # rotate ODM R into PATB orientation used by simple-ortho
    R = np.dot(R, np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]))
    # extract OPK from R - adapted from https://s3.amazonaws.com/mics.pix4d.com/KB/documents/Pix4D_Yaw_Pitch_Roll_Omega_to_Phi_Kappa_angles_and_conversion.pdf
    omega = np.arctan2(-R[1, 2], R[2, 2])
    phi = np.arcsin(R[0, 2])
    kappa =  np.arctan2(- R[0, 1], R[0, 0])
    return (omega, phi, kappa)

input_dir = r'./data/inputs/test_example3'
image_wild_card = r'*.JPG'
shots_file_path = Path(r'V:\Data\HomonimEgs\DJI_Phantom3_Sept_2019_Colorado\Source\odm_report\shots.geojson')
cam_pos_ori_file = 'camera_pos_ori.txt'

shots_dict = json.load(open(shots_file_path))
cam_pos_ori_dict = {}
for feature in shots_dict['features']:
    filename = feature['properties']['filename']
    pos = feature['properties']['translation']
    ori = feature['properties']['rotation']
    cam_pos_ori_dict[filename] = dict(pos=pos, ori=ori)

with open(Path(input_dir).joinpath(cam_pos_ori_file), 'w', newline='', encoding='utf-8') as out_file:
    writer = csv.writer(out_file, delimiter=' ')
    for src_path in Path(input_dir).glob(image_wild_card):
        if src_path.name not in cam_pos_ori_dict:
            print(f'Could not find {src_path.name} in {shots_file_path.name}')
            continue
        src_dict = cam_pos_ori_dict[src_path.name]
        # opk = hrp2opk(*np.rad2deg(src_dict['ori']))
        # src_dict['pos'][-1] += 15
        opk = np.degrees(angle_axis_to_opk(src_dict['ori']))
        writer.writerow([src_path.stem, *src_dict['pos'], *opk])
        # writer.writerow([src_path.stem, *src_dict['pos'], *src_dict['ori']])
        # writer.writerow([src_path.stem, *src_dict['pos'], *(np.rad2deg(src_dict['ori']))])
        # writer.writerow([src_path.stem, *src_dict['pos'], *(opk)])
