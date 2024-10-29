import os
import pandas as pd
from interpretable_ssl.evaluation.metrics import MetricCalculator


class MetricGenerator:
    def __init__(self, trainer) -> None:
        self.trainer = trainer
        self.all_metric_file = "all-semi-query-metrics.csv"

        self.all_metric_path = os.path.join(
            self.trainer.get_dump_path(), self.all_metric_file
        )
        print('metric generator for ', trainer.name)

    def clean_scib_df(self, df):
        if "Total" in df.columns:
            df = df.rename(columns={"Total": "scib total"})
        df_numeric = df.apply(pd.to_numeric, errors="coerce")

        # Drop rows where all values are NaN
        df_numeric = df_numeric.dropna(how="all")
        return df_numeric

    def load_metrics(self):
        # Get the dump folder path from the trainer
        dump_folder = self.trainer.get_dump_path()

        # Loop through all files in the specified folder
        for file in os.listdir(dump_folder):
            # Check if the file is a CSV and contains both 'semi' and 'query'
            if file.endswith(".csv") and "semi" in file and "query" in file:
                file_path = os.path.join(dump_folder, file)
                print(f"Loading file: {file_path}")
                # Load the CSV file into a DataFrame
                df = pd.read_csv(file_path)
                return self.clean_scib_df(df)
        return None

    def encode_query(self):
        try:
            self.trainer.init_scpoli()
        except:
            print(self.trainer.name, "cant init scpoli")
        return self.trainer.encode_query()

    def get_query_calculator(self):
        return MetricCalculator(
            self.trainer.query.adata,
            [self.encode_query()],
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

    def load_all_metrics(self):
        if os.path.exists(self.all_metric_path):
            return pd.read_csv(self.all_metric_path, index_col=0)
        return None

    def generate_metrics(self):
        all_metrics = self.load_all_metrics()
        if all_metrics is not None:
            return all_metrics

        metrics = self.load_metrics()

        calculator = self.get_query_calculator()
        knn = calculator.knn_results()
        linear = calculator.linear_results(50)

        all_metrics = [knn, linear]
        if not self.check_scgraph_exist(metrics):
            all_metrics.append(calculator.calculate_scgraph())

        if not self.check_scib_exist(metrics):
            scib_df = calculator.calculate_scib()
            all_metrics.append(self.clean_scib_df(scib_df))

        dfs = [df.reset_index(drop=True) for df in all_metrics]
        dfs.append(metrics)
        final_df = pd.concat(dfs, axis=1)
        final_df.index = [self.trainer.name]
        final_df.to_csv(self.all_metric_path)
        # final_df.index = self.trainer.name
        return final_df
