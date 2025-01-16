from constants import *

def generate_model_name(defaults, params):
   
    model_name_version = params.get('model_name_version', defaults.get('model_name_version'))

    """Generate a shortened job name based on the current parameters."""
    job_name_parts = []
    if 'prefix' in params:
        job_name_parts.append(params['prefix'])
        
    if model_name_version < 6:
        # Always include the augmentation type
        aug_type = params.get("augmentation_type", "community")
        # if aug_type:
        job_name_parts.append(
            f"{ABBREVIATIONS['augmentation_type']}_{str(aug_type)[:4]}"
        )
    else:
        job_name_parts.append(params.get('experiment_name', defaults.get('experiment_name')))
    # Include other parameters that differ from their defaults
    for key, val in params.items():
        if (
            key in ABBREVIATIONS
            and val is not None
            and val != defaults.get(key)
        ):
            if key == 'augmentation_type' and model_name_version < 6:
                continue
            if key == 'experiment_name' and model_name_version > 5:
                continue
            
            # Use the abbreviation for the key
            abbreviated_key = ABBREVIATIONS[key]
            # Shorten the value to a meaningful form (first 3-4 characters)
            if key == "experiment_name":
                max_length = len(val)
            else:
                max_length = 4
                
            if model_name_version > 5:
                if key == 'training_type':
                    max_length = len(val)
            shortened_val = str(val)[:max_length] if isinstance(val, str) else str(val)
            job_name_parts.append(f"{abbreviated_key}_{shortened_val}")
    
    # Join all parts to form the job name
    job_name = "_".join(job_name_parts)
    return job_name