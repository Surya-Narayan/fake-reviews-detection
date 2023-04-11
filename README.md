# Project: Fake Review Detection


## Preparing the Dataset
1. The dataset is too big to be uploaded at Github. Hence, we are uploading it in google drive and sharing the link here. Download the [YelpCSV folder](https://drive.google.com/drive/folders/1MexwmQTjEbz7UTQ2E_fPxLntE2ugeXtV?usp=share_link), [dataset_v1.csv](https://drive.google.com/file/d/1JzGEI0XdXk1l61vo3giNWUZykJzfRJKw/view?usp=share_link), and [review_feature_table.csv](https://drive.google.com/file/d/1QalYUu1pbkHglrF96GQw_0kL3thj8s17/view?usp=share_link) files, and place them into the DATA subdirectory. *Note: It's possible to run the program without downloading dataset_v1.csv and review_feature_table.csv beforehand. In this case, the program will re-calculate all features and carry out undersampling, which will result in a longer execution time.

```
.
├── DATA
     └── YelpCSV
            └── metadata.csv
            └── productIdMapping.csv
            └── reviewContent.csv
            └── reviewGraph.csv
            └── userIdMapping.csv
     └── dataset_v1.csv
     └── review_feature_table.csv

```

## Executing the Program
1. Make sure your virtual environment is active, and your current working directory is set to the main repository directory (e.g., path/to/dir/fake_review_detection). Then, execute the following command:
```bash
python3 ./CODE/main.py
```

## After Running the Program
Once the program has finished executing, all relevant metrics and visualizations will be generated and saved in the EVALUATIONS subdirectory.
