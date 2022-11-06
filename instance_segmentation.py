from detectron2.engine import DefaultPredictor
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances
from PIL import Image 
import PIL 
import cv2
import numpy as np


# register_coco_instances("dogs_segmentation_train", {}, 'train\_annotations.coco.json', 'train')
# dogs_metadata = MetadataCatalog.get('dogs_segmentation_train')

class Detector:

    register_coco_instances("dogs_segmentation_train", {}, 'train\_annotations.coco.json', 'train')

    def __init__(self, model_type = "instance_segmentation"):
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
        self.cfg.DATASETS.TRAIN = ('dogs_segmentation_train',)
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
        self.cfg.MODEL.WEIGHTS = 'detectron2_dogs_segmentation.pth'
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
        self.cfg.MODEL.DEVICE = "cpu" # cpu or cuda
        # self.dogs_metadata = MetadataCatalog.get('dogs_segmentation_train')

        self.predictor = DefaultPredictor(self.cfg)

    def onImage(self, imagePath):
        image = cv2.imread(imagePath)
        predictions = self.predictor(image)

        viz = Visualizer(image[:,:,::-1],metadata= MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]),scale=1.2)#instance_mode=ColorMode.IMAGE_BW)
        
        output = viz.draw_instance_predictions(predictions['instances'].to('cpu'))
        filename = 'result.jpg'
        cv2.imwrite(filename, output.get_image()[:,:,::-1])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

