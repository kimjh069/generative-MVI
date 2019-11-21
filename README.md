# Generative MVI

This repository will provide codes including network architecture, input data generation and Anti-specificity Loss proposed in our paper.

## 1) Network Architecture
<img src="./data/Network_G_D.png"></img>
The generator and discriminator of proposing network will be provided in the python file 'model.py'.

## 2) Training input data generation
Example image, guideline and mask are prepared in the 'data' directory.<br/>
<img src="./data/sample_image.jpg" width="200px" height="200px"></img>
<img src="./data/sample_guideline.jpg" width="200px" height="200px"></img>
<img src="./data/sample_mask.png" width="200px" height="200px"></img><br/>

You will be able to run the code to generate sample training input data.
```bash
python input_data_generation.py
```
Following images will be saved :
cropped ground truth image I_gt, guideline L_gt, color prior I_cp, no color mask M_cp, imperfect guideline ~L and no edge zone N.
<br/>
<img src="./data/result_crop_image_gt.png" width="200px" height="200px"></img>
<img src="./data/result_crop_guideline_gt.png" width="200px" height="200px"></img><br/>
<img src="./data/result_color_prior.png" width="200px" height="200px"></img>
<img src="./data/result_no_color_mask.png" width="200px" height="200px"></img>
<img src="./data/result_imperfect_guideline.png" width="200px" height="200px"></img>
<img src="./data/result_no_edge_zone.png" width="200px" height="200px"></img><br/>


## 3) Anti-specificity Loss
<img src="./data/Anti-specificity loss.gif"></img><br/>
The source code for the anti-specificity loss will be also available.
