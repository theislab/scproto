import os
import subprocess
import itertools
from constants import *
from interpretable_ssl.configs.defaults import *
import time


class ExperimentRunner:
    def __init__(
        self,
        template_file="run_template.sbatch",
        output_dir="slurm-job-out",
        conda_env="apex-env",
        sbatch_dir="./runs/sbatch_files/",
    ):
        self.template_file = template_file
        self.output_dir = output_dir
        self.conda_env = conda_env
        self.sbatch_dir = sbatch_dir
        # Set default parameters
        self.defaults = get_defaults()
        self.defaults.update(
            {
                "job_name": "default_job",
                "nodes": 1,
                "cpus_per_task": 10,
                "memory": "80G",
                "nice_value": 10000,
                "partition": "gpu_p",
                "qos": "gpu_normal",
                "gres": "gpu:1",
                "conda_env": self.conda_env,
                "num_prototypes": 300,
                "latent_dims": 8,
                "batch_size": 1024,
                "epsilon": 0.02,
                "cvae_loss_scaler": 0.0,
                "prot_decoding_loss_scaler": 0,
                "model_version": 1,
            }
        )
        self.defaults = {
            key: None if value == "" else value for key, value in self.defaults.items()
        }
        self.original_defaults = self.defaults.copy()
        self.qos_dict = {
            # "gpu_short": 2,
            "gpu_long": 3,
            "gpu_normal": 10,
            # "gpu_priority": 5,
        }

    def get_best_qos(self):
        """
        Returns the QoS with available capacity (i.e., where the number of running jobs is less than the max allowed).
        If no such QoS exists, return None.
        """
        for qos_name, max_jobs in self.qos_dict.items():
            running_jobs = count_jobs(qos_name)

            if running_jobs is not None and running_jobs < max_jobs:
                # Return the first QoS that has available capacity
                return qos_name

        # If no QoS has available capacity, return None
        return None

    def update_params(self, **kwargs):
        """Update experiment parameters with provided keyword arguments."""
        self.defaults.update(kwargs)

        # Set the output and error file paths based on the job_name
        job_name = self.defaults["job_name"]
        self.defaults["output_file"] = os.path.join(
            self.output_dir, job_name, "output.txt"
        )
        self.defaults["error_file"] = os.path.join(
            self.output_dir, job_name, "error.txt"
        )

    def generate_sbatch_content(self):
        """Generate the content for the SBATCH file based on the current parameters."""

        # Continuously try to get a valid QoS
        best_qos = None
        while not best_qos:
            best_qos = self.get_best_qos()
            if not best_qos:
                print("No available QoS. Retrying...")
                time.sleep(60)  # Wait 10 seconds before retrying

        print(best_qos)
        qos_resource = get_qos_resources(best_qos)

        # Update the defaults dictionary with the best QoS
        self.defaults["qos"] = best_qos
        self.defaults["memory"] = qos_resource["Memory"]
        self.defaults["cpus_per_task"] = qos_resource["CPU"]
        self.defaults['workers'] = qos_resource["CPU"] - 1
        # Read the SBATCH template file
        file_path = os.path.join(self.sbatch_dir, self.template_file)
        with open(file_path, "r") as file:
            sbatch_content = file.read()

        # Return the formatted content with the updated defaults
        return sbatch_content.format(**self.defaults)

    def run_experiment(self):
        """Generate the SBATCH file, save it, and submit the job."""
        job_name = self.defaults["job_name"]
        experiment_dir = os.path.join(self.output_dir, job_name)

        # Create directory for job output
        os.makedirs(experiment_dir, exist_ok=True)

        # Generate SBATCH content
        sbatch_content = self.generate_sbatch_content()

        # Write the SBATCH file
        sbatch_file = os.path.join(experiment_dir, f"run_{job_name}.sbatch")
        with open(sbatch_file, "w") as file:
            file.write(sbatch_content)

        # Submit the SBATCH job
        subprocess.run(["sbatch", sbatch_file])

        print(f"Submitted job: {job_name}")

    def run_multiple_experiments(self, item_to_test, submit=False):
        """Run or save multiple experiments by varying parameters according to the item_to_test."""
        keys, values = zip(*item_to_test.items())

        for value_combination in itertools.product(*values):
            params = dict(zip(keys, value_combination))

            # Generate job name automatically based on differences from defaults
            job_name = self.generate_job_name(params)
            current_jobs = get_slurm_job_names()
            if job_name in current_jobs:
                print(job_name, " already running")
                continue
            params["job_name"] = job_name
            # params["experiment_name"] = job_name

            self.update_params(**params)

            if submit:
                self.run_experiment()  # Submit the job
            else:
                self.save_sbatch_file()  # Only save the SBATCH file for review

    def generate_job_name(self, params):
        """Generate a shortened job name based on the current parameters."""
        job_name_parts = []

        # Always include the augmentation type
        aug_type = params.get("augmentation_type")
        if aug_type:
            job_name_parts.append(
                f"{ABBREVIATIONS['augmentation_type']}_{str(aug_type)[:4]}"
            )

        # Include other parameters that differ from their defaults
        for key, val in params.items():
            if (
                key != "augmentation_type"
                and key in ABBREVIATIONS
                and val is not None
                and val != self.original_defaults.get(key)
            ):
                # Use the abbreviation for the key
                abbreviated_key = ABBREVIATIONS[key]
                # Shorten the value to a meaningful form (first 3-4 characters)
                if key == "experiment_name":
                    max_length = len(val)
                else:
                    max_length = 4
                shortened_val = (
                    str(val)[:max_length] if isinstance(val, str) else str(val)
                )
                job_name_parts.append(f"{abbreviated_key}_{shortened_val}")

        # Join all parts to form the job name
        job_name = "_".join(job_name_parts)
        return job_name

    def save_sbatch_file(self):
        """Generate and save the SBATCH file without submitting it."""
        job_name = self.defaults["job_name"]
        experiment_dir = os.path.join(self.output_dir, "auto_generated", job_name)

        # Create directory for job output
        os.makedirs(experiment_dir, exist_ok=True)

        # Generate SBATCH content
        sbatch_content = self.generate_sbatch_content()

        # Write the SBATCH file
        sbatch_file = os.path.join(
            self.sbatch_dir, "auto_generated", f"run_{job_name}.sbatch"
        )
        with open(sbatch_file, "w") as file:
            file.write(sbatch_content)

        print(f"Saved SBATCH file: {sbatch_file}")


def evaluate_job_count(items_to_test):
    total_jobs = 0

    for idx, item_to_test in enumerate(items_to_test):
        keys, values = zip(*item_to_test.items())

        # Calculate the number of combinations for this particular item_to_test
        num_combinations = len(list(itertools.product(*values)))

        print(f"Item {idx + 1} will submit {num_combinations} jobs.")
        total_jobs += num_combinations

    print(f"Total number of jobs to be submitted: {total_jobs}")
    return total_jobs


def can_submit_new_job(max_jobs, user):
    """Check if the user can submit a new job based on the current number of running or pending jobs."""
    result = subprocess.run(["squeue", "-u", user, "-h"], stdout=subprocess.PIPE)
    current_jobs = len(result.stdout.decode("utf-8").splitlines())
    return current_jobs < max_jobs


def count_jobs(qos_name):
    try:
        # Construct the command
        command = f"squeue -u fatemehs.hashemig -O qos,state | grep {qos_name} | wc -l"

        # Execute the command and capture the output
        result = subprocess.check_output(command, shell=True, text=True)

        # Convert result to an integer
        running_jobs = int(result.strip())

        return running_jobs

    except subprocess.CalledProcessError as e:
        # Handle errors (e.g., if the command fails)
        print(f"Error executing command: {e}")
        return None


def get_slurm_job_names():
    user = "fatemehs.hashemig"
    try:
        # Run the 'squeue' command for the specified user and capture the output
        result = subprocess.run(
            ["squeue", "--user", user, "--format", "%.100j"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Check for errors
        if result.returncode != 0:
            print(f"Error: {result.stderr}")
            return []

        # Split the output into lines, strip leading and trailing whitespaces
        lines = [line.strip() for line in result.stdout.splitlines()]

        # Skip the header (first line) and return the remaining lines (job names)
        return lines[1:]  # Skip the first line if it contains the header

    except Exception as e:
        print(f"An error occurred: {e}")
        return []


def get_qos_resources(qos):
    # Dictionary mapping QoS to their respective memory and CPU limits
    qos_resources = {
        "gpu_short": {"CPU": 10, "Memory": "80G"},
        "gpu_normal": {"CPU": 20, "Memory": "160G"},
        "gpu_long": {"CPU": 28, "Memory": "240G"},
        "gpu_priority": {"CPU": 28, "Memory": "500G"},
        "interactive_gpu": {"CPU": 8, "Memory": "20G"},
        "interactive_gpu_short": {"CPU": 8, "Memory": "32G"},
    }

    # Return resources for the specified QoS
    return qos_resources.get(qos, "QoS not found")


if __name__ == "__main__":
    runner = ExperimentRunner("swav_template.sbatch")

    #  'scanpy_knn', 'community', 'cell_type'
    augmentaion_types = ["knn", "scanpy_knn", "community"]
    items_to_test = [
        {
            "dimensionality_reduction": [None],
            "prot_decoding_loss_scaler": [0, 5],
            "cvae_loss_scaler": [0, 0.0001],
            "k_neighbors": [7],
            "training_type": ["semi_supervised"],
            "fine_tuning_epochs": [150],
            "pretraining_epochs": [150],
            "augmentation_type": ["knn", 'community'],
            'num_prototypes': [50],
            "dataset_id": ["hlca"],
            
        },
        {
            "prot_decoding_loss_scaler": [0, 5],
            "cvae_loss_scaler": [0, 0.0001],
            "training_type": ["semi_supervised"],
            "fine_tuning_epochs": [150],
            "pretraining_epochs": [150],
            "augmentation_type": ["nb"],
            'num_prototypes': [50],
            "dataset_id": ["hlca"],
            
        },
    ]
    evaluate_job_count(items_to_test)

    transfer_lr_test = [
        {
            "prot_decoding_loss_scaler": [0],
            "cvae_loss_scaler": [0],
            "training_type": ["transfer_learning"],
            "fine_tuning_epochs": [5],
            "pretraining_epochs": [5],
            "augmentation_type": ["nb"],
            "experiment_name": ["test_transfer"],
            "pretrain_dataset_id": ["hlca"],
            "finetune_dataset_id": ["pbmc-immune"],
            "batch_size": [1025],
        }
    ]


    for item_to_test in items_to_test:
        # Run all experiments
        runner.run_multiple_experiments(item_to_test, True)
