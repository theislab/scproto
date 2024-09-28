from experiment_runner import ExperimentRunner  # Assuming ExperimentRunner is in a file named experiment_runner.py
from interpretable_ssl.trainers.swav import SwAV  # Import your SwAV model class
from interpretable_ssl.trainers.scpoli_original import OriginalTrainer  # Import your ScPoli model class (if needed)
import itertools
import os
import pandas as pd

class ExperimentEvaluator(ExperimentRunner):
    def __init__(self, model_type='swav', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_type = model_type

    def create_trainer(self, params, model_type):
        """Create a trainer instance based on the model type and parameters."""
        if model_type == 'swav':
            trainer = SwAV(
                num_prototypes=params.get('num_prototypes', self.original_defaults['num_prototypes']),
                latent_dims=params.get('latent_dims', self.original_defaults['latent_dims']),
                batch_size=params.get('batch_size', self.original_defaults['batch_size']),
                augmentation_type=params.get('augmentation_type', self.original_defaults['augmentation_type']),
                epsilon=params.get('epsilon', self.original_defaults['epsilon']),
                cvae_loss_scaler=params.get('cvae_loss_scaler', self.original_defaults['cvae_loss_scaler']),
                prot_decoding_loss_scaler=params.get('prot_decoding_loss_scaler', self.original_defaults['prot_decoding_loss_scaler']),
                model_version=params.get('model_version', self.original_defaults['model_version']),
                experiment_name=params.get('experiment_name', self.original_defaults['experiment_name']),
                k_neighbors=params.get('k_neighbors', self.original_defaults['k_neighbors']),
                dimensionality_reduction=params.get('dimensionality_reduction', self.original_defaults['dimensionality_reduction']), 
                training_type=params.get('training_type', self.original_defaults['training_type']),
            )
        elif model_type == 'scpoli':
            trainer = OriginalTrainer(latent_dims=params.get('latent_dims', self.original_defaults['latent_dims']),
            batch_size=params.get('batch_size', self.original_defaults['batch_size']),
            debug=params.get('debug', True),  # Assuming debug=True is a default
            experiment_name=params.get('experiment_name', self.original_defaults['experiment_name'])
        )
        else:
            raise ValueError("Unsupported model type: {}".format(model_type))
        
        trainer.name = self.generate_job_name(params, model_type)
        return trainer

    def generate_job_name(self, params, model_type):
        job_name = super().generate_job_name(params)
        return model_type + '_' + job_name
    
    def generate_trainers(self, item_to_test):
        """Generate a list of trainer instances based on the parameter grid."""
        trainer_list = []

        for model_type, model_params in item_to_test.items():
            keys, values = zip(*model_params.items())

            for value_combination in itertools.product(*values):
                params = dict(zip(keys, value_combination))

                # Generate a trainer based on the current parameters
                trainer = self.create_trainer(params, model_type)
                trainer_list.append(trainer)

        return trainer_list

def compare_trainers(trainers, file_type='ref'):
    """
    Compare trainers based on the metrics from their corresponding CSV files.

    Parameters:
    trainers (list): List of trainer objects.
    file_type (str): 'ref' or 'query' to specify which CSV file to load.

    Returns:
    pd.DataFrame: Styled DataFrame comparing the metric values across all trainers.
    """
    comparison_data = []
    for trainer in trainers:
        # Construct the path to the CSV file
        try:
            file_path = trainer.get_scib_file_path(file_type)
        except:
            file_path = f"{trainer.dump_path}/{file_type}-scib.csv"
        if file_path is None:
            file_path = f"{trainer.dump_path}/{file_type}-scib.csv"
        # Try to load the CSV file into a DataFrame
        try:
            df = pd.read_csv(file_path)
        except FileNotFoundError:
            print(f"File not found: {file_path}. Skipping this trainer.")
            continue
        
        # Extract the row with metric values (assuming it's the first row)
        metrics = df.iloc[0]
        
        # Determine the key to use for identifying the trainer
        key = getattr(trainer, 'name', None) or getattr(trainer, 'augmentation_type', None) or 'scpoli'
        
        # Create a dictionary to store the metrics with the key as an identifier
        metrics_dict = metrics.to_dict()
        metrics_dict['identifier'] = key
        
        # Append the metrics to the comparison data list
        comparison_data.append(metrics_dict)
    
    # Create a final DataFrame to compare all trainers
    final_df = pd.DataFrame(comparison_data)
    # Set the identifier as the index for better comparison
    final_df.set_index('identifier', inplace=True)
    
    # Convert all columns except the index to numeric and round to 2 decimal places
    final_df = final_df.apply(pd.to_numeric, errors='ignore').round(3)
    
    # Apply bold styling to the maximum value in each column
    def highlight_max(s):
        is_max = s == s.max()
        return ['font-weight: bold' if v else '' for v in is_max]
    
    # Apply rounding and remove unnecessary trailing zeros
    styled_df = final_df.style.apply(highlight_max, axis=0).format(lambda x: '{:.3f}'.format(x).rstrip('0').rstrip('.') if isinstance(x, float) else x)
    
    return styled_df

if __name__ == '__main__':
    evaluator = ExperimentEvaluator()

    item_to_test = {
        'swav': {
            'dimensionality_reduction': ['pca', None],
            'num_prototypes': [100, 300, 500],
            'latent_dims': [8, 16],
            'augmentation_type': ['knn', 'scanpy_knn', 'community']
        },
        'scpoli': {
            'latent_dims': [8, 16, 32],
            'batch_size': [512, 1024],
            'debug': [True, False]
        }
    }

    # Generate trainers based on parameter combinations
    trainers = evaluator.generate_trainers(item_to_test)

    # Evaluate the generated trainers
    evaluator.evaluate_trainers(trainers)

