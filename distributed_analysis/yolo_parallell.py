from ultralytics import YOLO
from ultralytics.engine.results import Results
from PIL import Image
import numpy as np
import os
import glob
import dask as da
from dask import array
from pathlib import Path
import random
from functools import partial
from threading import Lock

import threading

mutex = threading.Lock()

from timeit import default_timer as timer

random.seed()

@da.delayed
def segment_with_yolo(model, data):
    
    results = model.predict(source=np.ascontiguousarray(data), imgsz=512,show_boxes=False,show_labels=False, verbose=False)
    return results

def segment_wrapper(model, out_path, data, block_info=None):
    with mutex:
        #print("data shape: ", data.shape)
        result = segment_with_yolo(model,data)
        #print("hej hej")
        
    computed_result = result.compute()
        #end mutex here?
    if computed_result is None or computed_result[0].masks is None:
        return np.zeros(shape=(512,512,1), dtype=np.uint8)
    
    masks = computed_result[0].masks.data.cpu().numpy()
    shape = computed_result[0].masks.shape
    sh1 = shape[1]
    sh2 = shape[2]
    #all_masks = np.zeros(shape=(sh0,sh1,sh2), dtype=np.uint8)
    all_masks = np.zeros(shape=(512,512,1), dtype=np.uint8)

    for n in range(masks.shape[0]):
        mask = masks[n,:,:] * random.randint(1,255)
        mask = np.expand_dims(mask,axis=2)
        mask = mask.astype(np.uint8)
        if shape != (512,512): 
            all_masks[:sh1, :sh2,:] += mask     
        else:
            all_masks += mask
        # if not mask.shape == (512,512):
        #     mask.reshape(512,512)
        

    return all_masks

base = os.getcwd()
out_data_dir = Path(str(base) + "/output")
Path(out_data_dir).mkdir(parents=True, exist_ok=True)


model = YOLO(str(base) + "/latest_model.pt")
im = np.asarray(Image.open(str(base) + "/large.png").convert("RGB"))

large_image = da.array.from_array(im, chunks=(512,512,3))

print("starting...")
start = timer()
bound_f = partial(segment_wrapper, model, str(out_data_dir))
lazy_results = large_image.map_blocks(bound_f, dtype=large_image.dtype,chunks=(512,512,1))
result = lazy_results.compute(scheduler='single-threaded')
end = timer()
print("stopping: ",end - start)


save_im = Image.fromarray(result[:,:,0])
save_im.save("result_mask.png")
