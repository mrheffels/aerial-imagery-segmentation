import os
import sys
sys.path.append(os.getcwd())
from libs import inference
from libs import scoring




# dataset = 'dataset-sample'  #  0.5 GB download
dataset = 'dataset-medium' # 9.0 GB download
print("Running inference...")
sys.stdout.flush()
inference.run_inference(dataset)

# scores all the test images compared to the ground truth labels
score, predictions = scoring.score_predictions(dataset)
print(score)
sys.stdout.flush()
