"""The experimenter module is used to sample and run Barnoldswick WSIMOD.

This module is designed to be run in parallel as a jobarray. It generates
parameter samples and runs the Barnoldswick WSIMOD model for each sample. 
The results are saved to a csv file in a results directory.
"""
from __future__ import annotations

import argparse
import os
from collections import defaultdict
from pathlib import Path
from copy import deepcopy
import yaml

import wsimod.orchestration
import wsimod.orchestration.model
from wsimod.core import constants
from parameters.sa_parameters_bwick import get_sa_parameters, params_to_select
from sa_overrides import overrides
from multiprocessing import Pool

import pandas as pd
from SALib.sample import sobol 
from SALib.analyze import sobol as sobol_analyze
import numpy as np

# Set the number of threads to 1 to avoid conflicts with parallel processing
# for pysheds (at least I think that is what is happening)
os.environ['NUMBA_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import wsimod
os.environ['wsimod'] = "true"

SAMPLE_N = 300 # Adjust as needed

def formulate_salib_problem(parameters_to_select: 
                            list[str | dict] | None = None) -> dict:
    """Formulate a SALib problem for a sensitivity analysis.

    Args:
        parameters_to_select (list, optional): List of parameters to include in 
            the analysis, if a list entry is a dictionary, the value is the
            bounds, otherwise the bounds are taken from the parameters file.
            Defaults to None.

    Returns:
        dict: A dictionary containing the problem formulation.
    """
    # Set as empty by default
    parameters_to_select = [] if parameters_to_select is None else parameters_to_select

    # Get all parameters schema
    parameters = get_sa_parameters()
    names = []
    bounds = []
    dists = []
    groups = []

    for parameter in parameters_to_select:
        if isinstance(parameter, dict):
            bound = next(iter(parameter.values()))
            parameter = next(iter(parameter))
        else:
            bound = [parameters[parameter]['minimum'],
                      parameters[parameter]['maximum']]
        
        names.append(parameter)
        bounds.append(bound)
        dists.append(parameters[parameter].get('dist', 'unif')) # default to uniform if not specified
        groups.append(parameters[parameter]['category'])
    return {'num_vars': len(names), 'names': names, 'bounds': bounds,
            'dists': dists, 'groups': groups}

def generate_samples(N: int | None = None,
                     parameters_to_select: list[str | dict] = [],
                     seed: int  = 1,
                     groups: bool = False,
                     calc_second_order: bool = True) -> list[dict]:
    """Generate samples for a sensitivity analysis.

    Args:
        N (int, optional): Number of samples to generate. Defaults to None.
        parameters_to_select (list, optional): List of parameters to include in 
            the analysis, if a list entry is a dictionary, the value is the
            bounds, otherwise the bounds are taken from the parameters file.
            Defaults to [].
        seed (int, optional): Random seed. Defaults to 1.
        groups (bool, optional): Whether to sample by group, True, or by 
            parameter, False (significantly changes how many samples are taken). 
            Defaults to False.
        calc_second_order (bool, optional): Whether to calculate second order
            indices. Defaults to True.

    Returns:
        list: A list of dictionaries containing the parameter values.
    """
    problem = formulate_salib_problem(parameters_to_select)
    
    if N is None:
        N = SAMPLE_N #2 ** (problem['num_vars'] - 1) # this formula is only applicable when there are more parameters
    
    # If we are not grouping, we need to remove the groups from the problem to
    # pass to SAlib, but we retain the groups information for the output regardless
    problem_ = problem.copy()
    
    if not groups:
        del problem_['groups']
    
    # Sample
    param_values = sobol.sample(problem_, 
                                N, 
                                calc_second_order=calc_second_order, # total: split into 1st (just one para sensitvity) and 2nd (how sensitive to change in this parameter alongside with aonther parameter)
                                seed = seed) # same seed -> same samples, good for testing and debugging, change seed to get different samples
    # Store samples
    X = [
        {'param': y.split('::')[1], 'object': y.split('::')[0], 'value': z, 'iter': ix, 'group': x}
        for ix, params in enumerate(param_values)
        for x, y, z in zip(problem['groups'], problem['names'], params, strict=True)
    ]
    return X

def run_single_iteration(args):
    """Run a single model iteration. Designed for use with multiprocessing."""
    ix, params_, config_base, model_dir, results_fid = args

    # Skip if already done (allows resuming interrupted runs)
    output_file = results_fid / f'iter_{ix}.csv'
    if output_file.exists():
        print(f"Skipping iter {ix} — already done")
        return

    config = deepcopy(config_base)
    config["overrides"] = {"nodes": {}, "arcs": {}}

    for grp, param, obj, val in params_[["group", "param", "object", "value"]].itertuples(index=False, name=None):
        config = overrides(config, param, obj, val)

    config_fid = model_dir / f'{ix}_config.yml'
    with open(config_fid, 'w') as f:
        yaml.dump(config, f)

    model = wsimod.orchestration.model.Model()
    model.load(model_dir, config_name=config_fid.name)
    dates = [wsimod.orchestration.model.to_datetime(str(x)) for x in pd.date_range('2000-01-01', '2017-12-31')]
    model.dates = dates
    constants.FLOAT_ACCURACY = 1E-9
    flows, _, _, _ = model.run(dates=dates)
    flows = pd.DataFrame(flows)

    mean_flow = flows[flows['arc'] == '3607-land-to-3607-river']['flow'].mean()

    result_row = pd.DataFrame([{
        'iter': ix,
        'mean_flow': mean_flow,
        **params_.set_index('param_object').value.to_dict()
    }])
    result_row.to_csv(output_file, index=False)

    del model, flows, result_row
    config_fid.unlink()
    print(f"Iter {ix} done")

def process_parameters(jobid: int, # jobid=0 
                       nproc: int | None, # nproc=1
                       config_base: dict) -> tuple[dict[int, dict], Path]:
    """Generate and run parameter samples for the sensitivity analysis.

    This function generates parameter samples and runs the wsimod model
    for each sample. It is designed to be run in parallel as a jobarray. It 
    selects parameters values from the generated ones based on the jobid and 
    the number of processors. It copies the config file and passes these 
    parameters into wsimod via the manual overrides. Existing
    overrides that are not being sampled are retained, existing overrides that 
    are being sampled are overwritten by the sampled value.

    Args:
        jobid (int): The job id.
        nproc (int | None): The number of processors to use. If None, the number
            of samples is used (i.e., only one model is simulated).
        config_base (dict): The base configuration dictionary.

    Returns:
        dict[dict]: A dict (keys as models) of dictionaries containing the results.
        Path: The path to the inp file.
    """
    # Generate samples
    X = generate_samples(parameters_to_select=params_to_select())
    
    df = pd.DataFrame(X)
    gb = df.groupby('iter') # each parameter set is an iteration, so group by iteration to get the parameter sets for each model run
    n_iter = len(gb)
    print(f"{n_iter} samples created")
    flooding_results = {}

    nproc = nproc if nproc is not None else n_iter 

    # Assign jobs based on jobid
    if jobid >= nproc:
        raise ValueError("Jobid should be less than the number of processors.")
    job_idx = range(jobid, n_iter, nproc) 

    model_dir = Path(__file__).parent.parent / 'v2' / 'model'
    results_fid = Path(__file__).parent.parent / 'SA' / 'results'
    results_fid.mkdir(parents=True, exist_ok=True)

    # Build argument list for each iteration
    args_list = [
        (ix, gb.get_group(ix).assign(param_object=gb.get_group(ix)['object'] + '::' + gb.get_group(ix)['param']),
        config_base, model_dir, results_fid)
        for ix in job_idx
    ]

    n_cores = 3  # leave 7 cores free for my OS — adjust if needed
    with Pool(processes=n_cores) as pool:
        pool.map(run_single_iteration, args_list)

    flooding_results = {}
    return flooding_results

def save_results(jobid: int, results: dict[int, dict], address: Path) -> None:
    print("Results written incrementally during model runs.") # added this to not break the __main__ block
    

def parse_arguments() -> tuple[int, int | None, Path]:
    """Parse the command line arguments.

    Returns:
        tuple: A tuple containing the job id, number of processors, and the
            configuration file path.
    """
    parser = argparse.ArgumentParser(description='Process command line arguments.')
    parser.add_argument('--jobid', 
                        type=int, 
                        default=0, 
                        help='Job ID')
    parser.add_argument('--nproc', 
                        type=int, 
                        default=1,   
                        help='Number of processors')
    parser.add_argument('--config_path', 
                        type=Path, 
                        default=Path(__file__).parent.parent / 'v2' /\
                                    'model',
                        help='Configuration file path')

    args = parser.parse_args()

    return args.jobid, args.nproc, args.config_path


if __name__ == '__main__':
    jobid, nproc, config_path = parse_arguments()
    config_file = config_path / f'config.yml'

    # Load the configuration
    with open(config_file, 'r') as f:
        config_base = yaml.safe_load(f)

    # Sample and run
    flooding_results = process_parameters(jobid, nproc, config_base)

    # Save the results
    save_results(jobid, flooding_results, config_path)