# Generative Multi-view Inpainting for Image-based Rendering of Large Indoor Spaces

## 1) Input data generation
```bash
python input_data_generation.py
```

For example, prepare an example image, guideline and mask.<br/>

<img src="./data/sample_image.jpg" width="200px" height="200px"></img>
<img src="./data/sample_guideline.jpg" width="200px" height="200px"></img>
<img src="./data/sample_mask.png" width="200px" height="200px"></img><br/>

Following images will be saved :
cropped ground truth image I_gt, guideline L_gt, color prior I_cp, no color mask M_cp, imperfect guideline ~L and no edge zone N.
<br/>
<img src="./data/result_crop_image_gt.png" width="200px" height="200px"></img>
<img src="./data/result_crop_guideline_gt.png" width="200px" height="200px"></img>
<img src="./data/result_color_prior.png" width="200px" height="200px"></img>
<img src="./data/result_no_color_mask.png" width="200px" height="200px"></img>
<img src="./data/result_imperfect_guideline.png" width="200px" height="200px"></img>
<img src="./data/result_no_edge_zone.png" width="200px" height="200px"></img><br/>

## 2) Network Architecture


## Citation
