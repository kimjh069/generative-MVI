# Generative Multi-view Inpainting for Image-based Rendering of Large Indoor Spaces

This repository provides codes including network architecture and input data generation from the work "Generative Multi-view Inpainting for Image-based Rendering of Large Indoor Spaces" submitted to CVPR2020.

## 1) Network Architecture
<img src="./data/Network_G_D.png"></img>
The generator and discriminator of proposing network is provided in the python file 'model.py'.

## 2) Input data generation
Example image, guideline and mask are prepared in the 'data' directory.<br/>
<img src="./data/sample_image.jpg" width="200px" height="200px"></img>
<img src="./data/sample_guideline.jpg" width="200px" height="200px"></img>
<img src="./data/sample_mask.png" width="200px" height="200px"></img><br/>

You can run the code to generate sample data of inputs.
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


## Citation
