from constants import *

def generate_model_name(defaults, params):
   
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
            and val != defaults.get(key)
        ):
            # Use the abbreviation for the key
            abbreviated_key = ABBREVIATIONS[key]
            # Shorten the value to a meaningful form (first 3-4 characters)
            if key == "experiment_name":
                max_length = len(val)
            else:
                max_length = 4
            shortened_val = str(val)[:max_length] if isinstance(val, str) else str(val)
            job_name_parts.append(f"{abbreviated_key}_{shortened_val}")

    # Join all parts to form the job name
    job_name = "_".join(job_name_parts)
    return job_name