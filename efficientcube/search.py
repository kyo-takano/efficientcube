"""
This module provides a beam search algorithm for finding a solution path in a given environment.
The `beam_search` function in this file is designed for readability and reproducibility, and it may not be the most speed-optimized implementation.
"""

import time
import numpy as np
from copy import deepcopy
from contextlib import nullcontext
import torch

@torch.no_grad()
def beam_search(
        env,
        model,
        beam_width=1024,
        max_depth=1024, # Any arbitrary number above God's number will do
        skip_redundant_moves=True,
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        enable_fp16=False
    ):
    """
    Beam search algorithm to find a solution path based on a cumulative product of estimated probabilities.

    Args:
        env (object): A scrambled instance of the given environment.
        model (torch.nn.Module): PyTorch model used to predict the next move(s).
        beam_width (int, optional): Maximum number of candidate paths per depth. Defaults to 1024.
        max_depth (int, optional): Maximum depth of the search tree. Defaults to 1024. Any arbitrary number above God's number will do.
        skip_redundant_moves (bool, optional): If True, skip redundant moves like `R'` after `R`. Defaults to True.
        device (torch.device, optional): The device where the model should be evaluated. Defaults to 'cuda' if a CUDA-enabled GPU is available, otherwise 'cpu'.
        enable_fp16 (bool, optional): If True, enables mixed precision evaluation using Float16. Defaults to False.

    Returns:
        dict or None: A dictionary containing the result if solved or None if no solution is found.
        - If solved:
            {
                'solutions': List of moves forming the solution path,
                'num_nodes': Number of nodes expanded during the search,
                'time_taken': Time taken to find the solution
            }
        - If not solved: None
    """

    env_class_name = env.__class__.__name__
    assert env_class_name in ['Cube3','Puzzle15','LightsOut7']

    model.eval()
    with torch.cuda.amp.autocast(dtype=torch.float16) if enable_fp16 else nullcontext():
        # metrics
        num_nodes, time_0 = 0, time.time()
        candidates = [
            {"state":deepcopy(env.state), "path":[], "value":1.}
        ] # list of dictionaries

        for depth in range(max_depth+1):
            # TWO things at a time for every candidate: 1. check if solved & 2. add to batch_x
            batch_x = np.zeros((len(candidates), env.state.shape[-1]), dtype=np.int64)
            for i,c in enumerate(candidates):
                c_path, env.state = c["path"], c["state"]
                if c_path:
                    env.finger_ix(c_path[-1])
                    num_nodes += 1
                    if env.is_solved():
                        # Revert: array of indices => array of notations
                        c_path = [str(env.moves[i]) for i in c_path]
                        return {'solutions':c_path, "num_nodes":num_nodes, "times":time.time()-time_0}
                batch_x[i, :] = env.state

            # after checking the nodes expanded at the deepest    
            if depth==max_depth:
                print("Solution not found.")
                return None

            # make predictions with the trained DNN
            if len(candidates) < 2**17:
                batch_x = torch.from_numpy(batch_x).to(device)
                batch_p = model(batch_x)
                batch_p = torch.nn.functional.softmax(batch_p, dim=-1)
                batch_p = batch_p.detach().cpu().numpy()
            else:
                # split the batch so as to avoid 'CUDA out of memory' error.
                batch_p = [
                    torch.nn.functional.softmax( model( torch.from_numpy(batch_x_mini).to(device) ) ).to('cpu').detach().numpy() 
                    for batch_x_mini in np.split(batch_x, len(candidates)//(2**16))
                ]
                batch_p = np.concatenate(batch_p)

            # loop over candidates
            candidates_next_depth = []  # storage for the depth-level candidates storing (path, value, index).
            for i, c in enumerate(candidates):
                c_path = c["path"]
                value_distribution = batch_p[i, :] # output logits for the given state
                value_distribution *= c["value"] # multiply the cumulative probability so far of the expanded path

                for m, value in zip(env.moves_ix_inference, value_distribution): # iterate over all possible moves.
                    # predicted value to expand the path with the given move.

                    if env_class_name=='Cube3':
                        if c_path and skip_redundant_moves:
                            if m not in env.moves_ix_available_after[c_path[-1]]:
                                # Two mutually canceling moves
                                continue
                            elif len(c_path) > 1:
                                # if c_path[-2] == c_path[-1] == m:
                                if c_path[-2] == c_path[-1] == m:
                                    # Three subsequent moves that could be one
                                    continue
                                # elif (
                                #     c_path[-2][0] == m[0] and len(c_path[-2] + m) == 3
                                #     and c_path[-1][0] == env.pairing[m[0]]
                                # ):
                                elif (
                                    c_path[-2]//2 == m//2 and c_path[-2]%2 != m%2
                                    and c_path[-1]//2 == env.pairing_ix[m//2]
                                ):
                                    # Two mutually canceling moves sandwiching an opposite face move
                                    continue

                    elif env_class_name=='Puzzle15':
                        # Skip **physically** illegal moves (whether you like it or not)
                        target_loc = np.where(c['state'].reshape(4, 4) == 0)
                        if m==3: # if m=="R"
                            if not target_loc[1]: # zero_index (empty slot) on the left
                                continue
                        elif m==1: # if m=="D"
                            if not target_loc[0]: # on the top
                                continue
                        elif m==0: # if m=="U"
                            if target_loc[0]==3: # on the bottom
                                continue
                        elif m==2: # if m=="L"
                            if target_loc[1]==3: # on the right
                                continue

                        if c_path and skip_redundant_moves:
                            # Two cancelling moves
                            if env.pairing_ix[c_path[-1]] == m:
                                continue

                    elif env_class_name=='LightsOut7':
                        if skip_redundant_moves:
                            # logically meaningless operation
                            if m in c_path:
                                continue
                    else:
                        raise

                    # add to the next-depth candidates unless 'continue'd.
                    candidates_next_depth.append({
                        'state':deepcopy(c['state']),
                        "path": c_path+[m],
                        "value":value,
                    })

            # sort potential paths by expected values and renew as 'candidates'
            candidates = sorted(candidates_next_depth, key=lambda item: -item['value'])
            # if the number of candidates exceed that of beam width 'beam_width'
            candidates = candidates[:beam_width]
