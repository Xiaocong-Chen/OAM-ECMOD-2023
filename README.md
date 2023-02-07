# ECMOD
DASFAA2023:《Learning Enhanced Representations via Contrasting for Multi-view Outlier Detection》
***
In this work, we propose ECMOD for learning enhanced representations via contrasting for multi-view outlier detection. Technically, ECMOD leverages two channels, the reconstruction and the constraint view channels, to learn the multi-view data, respectively. The two channels enable ECMOD to capture the rich information better associated with outliers in a latent space due to fully considering the relationships among different views. Then, ECMOD integrates a contrastive technique between two groups of embeddings learned via the two channels, serving as an auxiliary task to enhance multi-view representations. Furthermore, we utilize neighborhood consistency to uniform the neighborhood structures among different views. It means that ECMOD has the ability to detect outliers in two or more views. Meanwhile, we develop an outlier score function based on different outlier types without clustering assumptions.
## Data
We provide the dataset source files from UCI (https://archive.ics.uci.edu/) and ODDS (http://odds.cs.stonybrook.edu/), which are unprocessed. The methods to process the data are given in the methods.py
## Environment
```
python:3.8
torch:1.11.0+cu113
scikit-learn:1.0.2
numpy:1.21.5
pandas:1.2.4
```
## Start
```
python main.py
```
