import argparse    
import yaml
import wandb
import os
import subprocess

def main():
    args = build_parser().parse_args()

    with open(args.config, 'r') as file:
        sweep_config = yaml.safe_load(file)

    sweep_id = wandb.sweep(sweep_config)

    
    model_dir = os.path.join('experiments', sweep_config['parameters']['model']['value'])
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
    
    log_dir = os.path.join('experiments', sweep_config['parameters']['model']['value'], f'sweep_{sweep_id}')
    os.mkdir(log_dir)

    result = subprocess.run(['condor_submit', f'SWEEP_ID={sweep_id}', 'condor/run_htc.job', '-batch-name', f'sweep-{sweep_id}'], capture_output=True, text=True)

    print("--- Standard Output ---")
    print(result.stdout.strip())
    if result.stderr:
        print("\n--- Standard Error ---")
        print(result.stderr.strip()) 

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser() 
    p.add_argument("config", help='filepath to sweep yaml config file', type=str)
    return p

if __name__ == "__main__":
    main()
