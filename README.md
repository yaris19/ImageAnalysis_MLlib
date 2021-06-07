## MLlib classifier

### Contributors
* Yaris van Thiel
* Julian Lerrick
* Bram Löbker
* Ricardo Tolkamp

### Dataset
This project uses the Gene expression dataset dataset from Kaggle. 
1. Download ```data_set_ALL_AML_independent.csv``` from https://www.kaggle.com/crawford/gene-expression to ```src/main/resources```
2. Download ```data_set_ALL_AML_train.csv``` from https://www.kaggle.com/crawford/gene-expression to ```src/main/resources```
Please save the dataset to ```src/main/resources```

### How to run
1. Follow this [guide](https://www.cloudera.com/tutorials/setting-up-a-spark-development-environment-with-scala.html) to set up a spark development with scala in IntelliJ
2. Use 64bit Java 1.8 (version 8)
3. Create a virtualenv for python and install packages:
    ```bash
    python3 -m venv env
    source env/bin/activate
    pip install -r requirements.txt
    ```
4. Run ```preprocess_data.py``` (when virtualenv is activated) to preprocess the data so that it can be used for machine learning
   ```bash
   python preprocess_data.py
   ``` 
5. Run the scala script by pressing the ```Run``` button in IntelliJ
