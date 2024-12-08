import os
import subprocess

def run_script(script_name):
    try:
        subprocess.run(['python', script_name], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running {script_name}: {e}")

if __name__ == "__main__":
    scripts = [
        'data_preprocessing.py',
        'model_training.py',
        'model_evaluation.py'
    ]

    for script in scripts:
        script_path = os.path.join(os.path.dirname(__file__), script)
        run_script(script_path)