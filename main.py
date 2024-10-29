import sys
import argparse

def get_trainer(model_name, parser):
    if model_name == "swav":
        from interpretable_ssl.trainers.swav import SwAV
        return SwAV(parser = parser)
    elif model_name == "scpoli":
        from interpretable_ssl.trainers.scpoli_original import OriginalTrainer
        return OriginalTrainer(parser = parser)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def main():
    if len(sys.argv) < 2:
        raise ValueError("Model name must be provided as the first argument.")
    
    model_name = sys.argv[1]
    print(model_name)
    sys.argv = sys.argv[1:]
    parser = argparse.ArgumentParser(description=f"{model_name} Trainer Parameters")
    
    trainer = get_trainer(model_name, parser)
    trainer.setup()
    print(trainer.dump_path, " has been set up")
    trainer.run()

if __name__ == "__main__":
    print('-----main started----')
    main()
