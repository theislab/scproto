import os
import pandas as pd
from interpretable_ssl.evaluation.metrics import MetricCalculator


class MetricGenerator:
    def __init__(self, trainer) -> None:
        self.trainer = trainer
        
        # if self.trainer.finetuning:
        #     self.all_metric_file = "all-semi-query-metrics.csv"
        # else:
        #     self.all_metric_file = "all-pretrained-query-metrics.csv"
            
        # self.all_metric_path = os.path.join(
        #     self.trainer.get_dump_path(), self.all_metric_file
        # )
        print('metric generator for ', trainer.name)

    def clean_scib_df(self, df):
        if "Total" in df.columns:
            df = df.rename(columns={"Total": "scib total"})
        df_numeric = df.apply(pd.to_numeric, errors="coerce")

        # Drop rows where all values are NaN
        df_numeric = df_numeric.dropna(how="all")
        return df_numeric

    def load_metrics(self, split='query'):
        # Get the dump folder path from the trainer
        dump_folder = self.trainer.get_dump_path()

        # Loop through all files in the specified folder
        for file in os.listdir(dump_folder):
            # Check if the file is a CSV and contains both 'semi' and 'query'
            if file.endswith(".csv") and "semi" in file and split in file:
                file_path = os.path.join(dump_folder, file)
                print(f"Loading file: {file_path}")
                # Load the CSV file into a DataFrame
                df = pd.read_csv(file_path)
                return self.clean_scib_df(df)
        return None

    def encode_query(self):
        # try:
        #     self.trainer.init_scpoli()
        # except:
        #     print(self.trainer.name, "cant init scpoli")
        return self.trainer.encode_query()

    def get_adata_calculator(self, adata, retrain_epochs=0):
        
        return MetricCalculator(
            adata,
            [self.trainer.encode_adata(adata, retrain_epochs=retrain_epochs)],
            self.trainer.get_dump_path(),
            [self.trainer.name],
            self.trainer.get_dump_path(),
        )

    def check_scgraph_exist(self, metrics):

        if metrics is not None:
            # Check if the 'Corr-PCA' column exists in the DataFrame
            if "Corr-PCA" in metrics.columns:
                return True  # Return True if the column exists

        # Return False if no file with 'Corr-PCA' column is found
        return False

    def check_scib_exist(self, metrics):
        if metrics is not None:
            if "Batch correction" in metrics.columns:
                return True
        return False

    def get_metric_path(self, split, retrain_epochs=0):
        if self.trainer.finetuning:
            all_metric_file = f"all-semi-{split}-metrics"
        else:
            all_metric_file = f"all-pretrained-{split}-metrics"
        if retrain_epochs > 0:
            all_metric_file += f"-retrain-{retrain_epochs}"
        return os.path.join(
            self.trainer.get_dump_path(), all_metric_file + '.csv'
        )
    
    def load_all_metrics(self, split, retrain_epochs=0):
        path = self.get_metric_path(split, retrain_epochs)
        if os.path.exists(path):
            df = pd.read_csv(path, index_col=0)
            df.index = [self.trainer.name]
            return df
        return None

    def get_adata(self, split):
        
        if split == 'query':
            ds = self.trainer.query
        elif split == 'ref':
            ds = self.trainer.ref
        elif split == 'all':
            ds = self.trainer.dataset
        else: 
            raise ValueError('split should be query, ref or all')
        
        return ds.adata
    def generate_metrics(self, split='query', retrain_epochs=0):
        all_metrics = self.load_all_metrics(split, retrain_epochs)
        if all_metrics is not None:
            return all_metrics

        metrics = self.load_metrics(split)

        calculator = self.get_adata_calculator(self.get_adata(split), retrain_epochs)
        knn = calculator.knn_results()
        linear = calculator.linear_results(50)

        all_metrics = [knn, linear]
        if not self.check_scgraph_exist(metrics):
            all_metrics.append(calculator.calculate_scgraph())

        if (not self.check_scib_exist(metrics)) or retrain_epochs > 0:
            scib_df = calculator.calculate_scib()
            all_metrics.append(self.clean_scib_df(scib_df))

        dfs = [df.reset_index(drop=True) for df in all_metrics]
        dfs.append(metrics)
        final_df = pd.concat(dfs, axis=1)
        final_df.index = [self.trainer.name]
        final_df.to_csv(self.get_metric_path(split, retrain_epochs))
        # final_df.index = self.trainer.name
        return final_df
