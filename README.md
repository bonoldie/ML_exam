# Machine learning & artificial intelligence project - Computer engineering for robotics and smart industries - A.A. 2023/24

__Comparison between classification algorithms__ 

Authors: Bonoldi Enrico, Luca Ponti   

dataset: [food-101-tiny](https://www.kaggle.com/datasets/kmader/food41)

## Notebooks

Classic (feature extraction based on PCA and LDA): `nb_classic.ipynb`
  
VGG (feature extraction based on VGG16): `nb_vgg.ipynb`

## Quickstart 

### Linux 

Create and activate the conda environment

```bash
conda env create -f environment.yml && conda activate $(head -1 environment.yml | cut -d':' -f 2)
``` 

Launch the notebook 

```bash
jupyter notebook
```

or use the notebook in vs-code with the jupyter extension