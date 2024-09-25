from ultralytics import YOLO
#import config
from PIL import Image
import numpy as np
import os
import glob
import dask as da
from dask import array
from pathlib import Path
import random
import copy
from functools import partial
from threading import Lock
from skimage import io
#import distributed_map

random.seed()

mutex = Lock()

def segment_with_yolo(model, out_path, data):
    # with mutex:
    #     print("Copying model")
    #     model_copy = copy.copy(model)
    
    results = model(data,task="segment", show_boxes=False,show_labels=False,imgsz=config.IMG_SIZE)
    results[0].save(out_path + "/" + config.TEST_IMAGE_RESULT_FILE_NAME)
    return

def start():
    #TODO: read path from confg? And if it does not exist, do magic below
    file_dir = Path(__file__).parent.resolve()
    base = file_dir
    out_data_dir = Path(str(base) + "/output")
    #base = os.getcwd()
    #img_dir = f"{config.IMG_SIZE}x{config.IMG_SIZE}"
    Path(out_data_dir).mkdir(parents=True, exist_ok=True)
    test_data_path = str(base)
    #test_images = glob.glob(test_data_path + "grayscale" + "/*.png")
    test_images_2 = glob.glob(test_data_path + "/../512x512" + "/*.png")
    test_image = random.choice(test_images_2)
    input = Image.open(test_image)
    im = np.asarray(input.convert("RGB"))

    #imtest = np.asarray(Image.open(test_images_2[1])).swapaxes(0,1)

    #large_image = array.from_array(im, chunks=(512,512))

    model = YOLO(str(base) + "/../latest_model.pt")

    result = model.predict(source=im, imgsz=512,show_boxes=False,show_labels=False)
    if len(result) == 0:
        print("No result")
        return 
    
    masks = result[0].masks.data.cpu().numpy()
    #masks = np.transpose(masks,axes=(2,1,0))
    all_masks = np.zeros(shape=(512,512), dtype=np.uint8)
    for n in range(masks.shape[0]):
        mask = masks[n,:,:] * random.randint(1,255)
        print(f"unique: {np.unique(mask)}")
        mask = mask.astype(np.uint8)
        all_masks = np.add(all_masks,mask) 
        

    maskim = Image.fromarray(all_masks)
#    mask.convert("RGB")
    #all_zeros = not np.any(mask)
    input.paste(maskim.convert("rgb"),mask=maskim)
    input.show()
    #io.imshow(all_masks,)
    #io.show()
    #maskim.save(Path(test_image).stem + "_mask.png")
    #print(f"hej {all_zeros}")
    #bound_f = partial(segment_with_yolo, model, str(out_data_dir))
    #distributed_map.distributed_map_and_merge(large_image, bound_f)


# results = model(test_image,task="segment", show_boxes=False,show_labels=False,imgsz=config.IMG_SIZE)
# segments = results[0].masks
# results[0].show()
# results[0].save(str(base) + "/" + config.TEST_IMAGE_RESULT_FILE_NAME)

if __name__ ==  '__main__':
    start()