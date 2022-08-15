import argparse
import layoutparser as lp
from PIL import Image
import cv2
import numpy as np
import pandas as pd 
import tensorflow as tf
import os
from pdf2image import convert_from_path
from paddleocr import PaddleOCR, draw_ocr

model = lp.PaddleDetectionLayoutModel(config_path="lp://PubLayNet/ppyolov2_r50vd_dcn_365e_publaynet/config",
                                threshold=0.5,
                                label_map={0: "Text", 1: "Title", 2: "List", 3:"Table", 4:"Figure"},
                                enforce_cpu=False,
                                enable_mkldnn=True)#math kernel library


def make_parser():
  parser = argparse.ArgumentParser("PDF Extractor!")

  parser.add_argument(
      "--input_path", default="/content/drive/...", help="path to pdfs"
  )
  parser.add_argument(
      "--output_path", default="/content/drive/...", help="path to output csvs"
  )

  return parser
def intersection(box_1, box_2):
  return [box_2[0], box_1[1],box_2[2], box_1[3]]

def iou(box_1, box_2):
  x_1 = max(box_1[0], box_2[0])
  y_1 = max(box_1[1], box_2[1])
  x_2 = min(box_1[2], box_2[2])
  y_2 = min(box_1[3], box_2[3])

  inter = abs(max((x_2 - x_1, 0)) * max((y_2 - y_1), 0))
  if inter == 0:
      return 0
      
  box_1_area = abs((box_1[2] - box_1[0]) * (box_1[3] - box_1[1]))
  box_2_area = abs((box_2[2] - box_2[0]) * (box_2[3] - box_2[1]))
  
  return inter / float(box_1_area + box_2_area - inter)

def get_csv(csv_path,im,pg,tab,x_1,x_2,y_1,y_2):

  cv2.imwrite('ext_im.jpg', im[y_1:y_2,x_1:x_2])
  ocr = PaddleOCR(lang='en')
  image_path = 'ext_im.jpg'
  image_cv = cv2.imread(image_path)
  image_height = image_cv.shape[0]
  image_width = image_cv.shape[1]
  output = ocr.ocr(image_path)

  boxes = [line[0] for line in output]
  texts = [line[1][0] for line in output]
  probabilities = [line[1][1] for line in output]

  image_boxes = image_cv.copy()

  for box,text in zip(boxes,texts):
    cv2.rectangle(image_boxes, (int(box[0][0]),int(box[0][1])), (int(box[2][0]),int(box[2][1])),(0,0,255),1)
    cv2.putText(image_boxes, text,(int(box[0][0]),int(box[0][1])),cv2.FONT_HERSHEY_SIMPLEX,1,(222,0,0),1)

  im = image_cv.copy()

  horiz_boxes = []
  vert_boxes = []

  for box in boxes:
    x_h, x_v = 0,int(box[0][0])
    y_h, y_v = int(box[0][1]),0
    width_h,width_v = image_width, int(box[2][0]-box[0][0])
    height_h,height_v = int(box[2][1]-box[0][1]),image_height

    horiz_boxes.append([x_h,y_h,x_h+width_h,y_h+height_h])
    vert_boxes.append([x_v,y_v,x_v+width_v,y_v+height_v])

    cv2.rectangle(im,(x_h,y_h), (x_h+width_h,y_h+height_h),(0,0,255),1)
    cv2.rectangle(im,(x_v,y_v), (x_v+width_v,y_v+height_v),(0,255,0),1)

  horiz_out = tf.image.non_max_suppression(
      horiz_boxes,
      probabilities,
      max_output_size = 1000,
      iou_threshold=0.1,
      score_threshold=float('-inf'),
      name=None
  )

  horiz_lines = np.sort(np.array(horiz_out))
  im_nms = image_cv.copy()

  for val in horiz_lines:
    cv2.rectangle(im_nms, (int(horiz_boxes[val][0]),int(horiz_boxes[val][1])), (int(horiz_boxes[val][2]),int(horiz_boxes[val][3])),(0,0,255),1)

  vert_out = tf.image.non_max_suppression(
      vert_boxes,
      probabilities,
      max_output_size = 1000,
      iou_threshold=0.1,
      score_threshold=float('-inf'),
      name=None
  )
  vert_lines = np.sort(np.array(vert_out))

  for val in vert_lines:
    cv2.rectangle(im_nms, (int(vert_boxes[val][0]),int(vert_boxes[val][1])), (int(vert_boxes[val][2]),int(vert_boxes[val][3])),(255,0,0),1)

  out_array = [["" for i in range(len(vert_lines))] for j in range(len(horiz_lines))]

  unordered_boxes = []

  for i in vert_lines:
    unordered_boxes.append(vert_boxes[i][0])

  ordered_boxes = np.argsort(unordered_boxes)

  for i in range(len(horiz_lines)):
    for j in range(len(vert_lines)):
      resultant = intersection(horiz_boxes[horiz_lines[i]], vert_boxes[vert_lines[ordered_boxes[j]]] )

      for b in range(len(boxes)):
        the_box = [boxes[b][0][0],boxes[b][0][1],boxes[b][2][0],boxes[b][2][1]]
        if(iou(resultant,the_box)>0.1):
          out_array[i][j] = texts[b]

  out_array = np.array(out_array)
  pd.DataFrame(out_array).to_csv(csv_path+str(pg)+'_'+str(tab)+'.csv')
  return None

def create_tables(pdf_dir,out_dir):
  for ind in os.listdir(pdf_dir):
    print(ind[:-4])

    current_path='pages/'+ind[:-4]+'/'
    csv_path=out_dir#'/content/drive/MyDrive/food_composition/outputs/Google/'
    print('current_path--->',current_path)
    print('csv_path------->',csv_path)
    
    #check the number of csvs outputted and also do some postprocessing
    try:
      os.mkdir(current_path)
      os.mkdir(csv_path)
    except:
      print('exists')
    print('->',pdf_dir+ind)
    try:
      images = convert_from_path(pdf_dir+ind)
    except:
      print('not a pdf!')
    k=1
    for i in range(len(images)):
      images[i].save(current_path+'_'+str(i)+'.jpg', 'JPEG')
      image = cv2.imread(current_path+'_'+str(i)+'.jpg')
      image = image[..., ::-1]
      # detect
      layout = model.detect(image)
      # load model
      for l in layout:
        if l.type == 'Table':
          x_1 = int(l.block.x_1)
          y_1 = int(l.block.y_1)
          x_2 = int(l.block.x_2)
          y_2 = int(l.block.y_2)
          print('i is======',i)
          print('k is======',k)
          im=cv2.imread(current_path+'_'+str(i)+'.jpg')
          try:
            get_csv(csv_path,im,ind[:-4],k,x_1,x_2,y_1,y_2)
          except:
            print('error!')
          k+=1
  print('\n\n Extraction Completed !!!')

def pdf(filename):
  if filename[-3:]=='pdf':
    return True
  return False

def main(args):
  print(args.input_path,args.output_path)
  try:
    os.mkdir('pages/')
  except:
    print('pages directory already exists')
  create_tables(args.input_path,args.output_path)

if __name__ == "__main__":
  args = make_parser().parse_args()
  main(args)