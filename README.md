# Generative Multi-view Inpainting for Image-based Rendering of Large Indoor Spaces

## 1) Input data generation

```bash
python input_data_generation.py
```

For example, prepare an example image and guideline.<br/>
<img src="./data/result_crop_image_gt.png" width="200px" height="200px"></img>
<img src="./data/result_crop_guideline_gt.png" width="200px" height="200px"></img><br/>

Following images will be saved :
cropped ground truth image $\mathbf{I}_{gt}$ and guideline $\mathbf{L}_{gt}$, color prior $\mathbf{I}_{cp}$, no color mask $\mathbf{M}_{cp}$, imperfect guideline $\mathbf{\widetilde{L}}$, no edge zone $\mathbf{N}$.<br/>
<img src="./data/result_color_prior.png" width="200px" height="200px"></img>
<img src="./data/result_no_color_mask.png" width="200px" height="200px"></img>
<img src="./data/result_imperfect_guideline.png" width="200px" height="200px"></img>
<img src="./data/result_no_edge_zone.png" width="200px" height="200px"></img><br/>

<!--[image3](./data/result_color_prior.png)
![image4](./data/result_imperfect_guideline.png)
![image5](./data/result_crop_guideline_gt.png)-->

## Citation
