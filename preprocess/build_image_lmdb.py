import os
import sys
import numpy as np
import json
import collections
from PIL import Image
import matplotlib.pyplot as plt
import lmdb
import math

sys.path.append('codes/Matterport3DSimulator/build/')
import MatterSim


# Simulator image parameters
WIDTH = 640
HEIGHT = 480
VFOV = 60

scan_data_dir = 'datasets/Matterport3D/v1_unzip_scans'
connectivity_dir = 'datasets/R2R/connectivity'

sim = MatterSim.Simulator()
sim.setDatasetPath(scan_data_dir)
sim.setNavGraphPath(connectivity_dir)
sim.setPreloadingEnabled(True)
sim.setCameraResolution(WIDTH, HEIGHT)
sim.setCameraVFOV(math.radians(VFOV))
sim.setDiscretizedViewingAngles(True)
sim.setBatchSize(1)
sim.initialize()

viewpoint_ids = []
with open(os.path.join(connectivity_dir, 'scans.txt')) as f:
    scans = [x.strip() for x in f]
for scan in scans:
    with open(os.path.join(connectivity_dir, '%s_connectivity.json'%scan)) as f:
        data = json.load(f)
        viewpoint_ids.extend([(scan, x['image_id']) for x in data if x['included']])
print('Loaded %d viewpoints' % len(viewpoint_ids))


NEWHEIGHT = 248
NEWWIDTH = int(WIDTH / HEIGHT * NEWHEIGHT)
print(NEWHEIGHT, NEWWIDTH)

data_size_per_img = np.random.randint(255, size=(NEWHEIGHT, NEWWIDTH, 3), dtype=np.uint8).nbytes
print(data_size_per_img, 36*data_size_per_img*len(viewpoint_ids))

lmdb_path = 'datasets/R2R/features/panoimages.lmdb'

env = lmdb.open(lmdb_path, map_size=int(1e12))


for i, viewpoint_id in enumerate(viewpoint_ids):
    scan, vp = viewpoint_id
    if i % 100 == 0:
        print(i, scan, vp)
    
    key = '%s_%s' % (scan, vp)
    key_byte = key.encode('ascii')
    
    txn = env.begin(write=True)
        
    images = []
    for ix in range(36):
        if ix == 0:
            sim.newEpisode([scan], [vp], [0], [math.radians(-30)])
        elif ix % 12 == 0:
            sim.makeAction([0], [1.0], [1.0])
        else:
            sim.makeAction([0], [1.0], [0])
        state = sim.getState()[0]
        assert state.viewIndex == ix
        image = np.array(state.rgb, copy=True) # in BGR channel
        image = Image.fromarray(image[:, :, ::-1]) #cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # resize
        image = image.resize((NEWWIDTH, NEWHEIGHT), Image.ANTIALIAS)
        image = np.array(image)
        images.append(image)
    images = np.stack(images, 0)
    
    txn.put(key_byte, images)
    txn.commit()

env.close()
