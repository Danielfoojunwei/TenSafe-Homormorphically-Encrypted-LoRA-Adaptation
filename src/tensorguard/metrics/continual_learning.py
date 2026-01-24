from typing import List, Optional, Union
import numpy as np

def compute_cl_metrics(eval_matrix: List[List[float]], task_baselines: Optional[List[float]] = None):
    """
    Computes standard Continual Learning metrics from an evaluation matrix R.
    R[t, i] = performance on task i after training on task t.
    
    Args:
        eval_matrix: NxN matrix where N is the number of tasks.
        task_baselines: Optional baseline performance for each task i before any training.
        
    Returns:
        dict: {avg_accuracy, forgetting_mean, forgetting_max, bwt, fwt}
    """
    R = np.array(eval_matrix)
    n = R.shape[0]
    
    # 1. Average Accuracy (ACC)
    # ACC = (1/n) * sum_{i=1}^n R[n, i]
    avg_accuracy = np.mean(R[n-1, :])
    
    # 2. Forgetting (F)
    # f_{t, i} = max_{j in {1...t-1}} R[j, i] - R[t, i]
    # We simplify to f_i = R[curr, i] - R[final, i] if j=i is the peak (standard for PEFT)
    task_forgetting = []
    for i in range(n - 1): # Last task has no forgetting yet
        peak_perf = np.max(R[i:n-1, i])
        final_perf = R[n-1, i]
        task_forgetting.append(peak_perf - final_perf)
    
    forgetting_mean = np.mean(task_forgetting) if task_forgetting else 0.0
    forgetting_max = np.max(task_forgetting) if task_forgetting else 0.0
    
    # 3. Backward Transfer (BWT)
    # BWT = (1 / (n-1)) * sum_{i=1}^{n-1} (R[n, i] - R[i, i])
    if n > 1:
        bwt_terms = [R[n-1, i] - R[i, i] for i in range(n - 1)]
        bwt = np.mean(bwt_terms)
    else:
        bwt = 0.0
        
    # 4. Forward Transfer (FWT)
    # FWT = (1 / (n-1)) * sum_{i=2}^n (R[i-1, i] - b_i)
    fwt = 0.0
    if n > 1 and task_baselines:
        fwt_terms = [R[i-1, i] - task_baselines[i] for i in range(1, n)]
        fwt = np.mean(fwt_terms)
    elif n > 1:
        # Proxy FWT: comparison to R[0, i] if R[0, i] is treated as "pre-train"
        fwt_terms = [R[i-1, i] - R[0, i] for i in range(1, n)]
        fwt = np.mean(fwt_terms)

    return {
        "avg_accuracy": float(avg_accuracy),
        "forgetting_mean": float(forgetting_mean),
        "forgetting_max": float(forgetting_max),
        "bwt": float(bwt),
        "fwt": float(fwt),
        "accuracy_init": [float(R[i, i]) for i in range(n)],
        "accuracy_final": [float(R[n-1, i]) for i in range(n)]
    }
