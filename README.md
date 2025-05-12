# Customer Segmentation Project

![GitHub last commit](https://img.shields.io/github/last-commit/JasminaPZ/Customer_Segmentation_Project)
![GitHub repo size](https://img.shields.io/github/repo-size/JasminaPZ/Customer_Segmentation_Project)
![GitHub issues](https://img.shields.io/github/issues/JasminaPZ/Customer_Segmentation_Project)
![GitHub stars](https://img.shields.io/github/stars/JasminaPZ/Customer_Segmentation_Project?style=social)
![GitHub forks](https://img.shields.io/github/forks/JasminaPZ/Customer_Segmentation_Project?style=social)

This project uses the [Online Retail UCI Machine Learning Repository dataset](https://archive.ics.uci.edu/ml/datasets/online+retail) to perform customer segmentation. It demonstrates the full data pipeline, including preprocessing, feature engineering, clustering, and evaluation.

## Project Structure   

Customer_Segmentation_Project/    
│
├── README.md    
├── .gitignore    
│          
├── data/      
│ └── Online_Retail_UCI.csv      
│        
├── notebooks/        
│ └── Customer_Segmentation_Analysis.ipynb      
│        
├── src/        
│ ├── data_preprocessing.py      
│ ├── feature_engineering.py      
│ ├── clustering.py      
│ └── evaluation.py      
│        
└── images/      
├── elbow_plot.png      
├── dendrogram.png      
├── tsne_plot.png      
└── pca_clusters.png        

## Getting Started

To run this project locally, follow these steps:

### 1. Clone the repository

```bash
git clone https://github.com/JasminaPZ/Customer_Segmentation_Project.git
cd Customer_Segmentation_Project  
```
### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate         # For Mac/Linux
venv\Scripts\activate.bat        # For Windows
```
### 3. Install the required packages

```bash
pip install -r requirements.txt
```
### 4. Run the Jupyter notebook

Make sure you have Jupyter Notebook installed. Then, run:

```bash
jupyter notebook notebooks/Customer_Segmentation_Analysis.ipynb
```
### 5. Running the Python scripts (optional)

You can also run the Python scripts in the **src/** directory:

```bash
python src/data_preprocessing.py
python src/feature_engineering.py
python src/clustering.py
python src/evaluation.py
```
