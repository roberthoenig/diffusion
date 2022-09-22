# Diffusion Semester Project

## Repository layout

* `download_3rcsan.py`: Self-documented script to download (parts of)
  the [3RScan](https://github.com/WaldJohannaU/3RScan) dataset.

* `bounding_boxes.ipynb`: Notebook to download bounding box data from
  3RScan, parse the data into a dataset of scenes and visualize the scenes.

* `bbox_diffusion.ipynb`: Notebook to train a diffusion model
  on the bounding boxes dataset and generate bounding box scenes.

* `toy_example.py`: Educational diffusion model that learns to generate
  Swiss rolls.