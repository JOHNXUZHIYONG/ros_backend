#!/usr/bin/env python3
# encoding: utf-8

"""
Helper function for generating JPEG plots to put in the PDF.

The plot generator function must be supplied by the core logic from Ernest.

Author: Ernest
"""
import importlib
from multiprocessing import Lock
from shutil import rmtree
from typing import Any, List, Iterable, Dict

from helper import TOP_LEVEL_PACKAGE, SUBPACKAGE, CONFIG_DATA, os, \
    NUM_RESULTS_DIR, INPUT_EX, OUTPUT_EX, logging

lock = Lock()  # To prevent image dump conflict.


# noinspection PyUnusedLocal
def generate_plots(figures_to_plot: Iterable[str],
                   out_dump_directory: str,
                   input_variables: Dict[str, List[float]]) -> None:
    """
    Helper function to generate all the figures.

    figures_to_plot:  A list of figures the user wants plotted.

    out_dump_directory: Path where the figures are to be dumped.

    input_variables: User supplied parameters by slider input

    If the output dump directory does not exist, it is created.
    """

    def single_plot_generator(figure_name: str,
                              input_params: Dict[str, List[float]]) -> None:
        """
        Generates a single plot. Assumes the input data is available.

        If the directory does not exist, it is created. 
        """
        in_file: str = os.path.join(NUM_RESULTS_DIR,
                                    f'{figure_name}.{INPUT_EX}')
        out_file: str = os.path.join(out_dump_directory,
                                     f'{figure_name}.{OUTPUT_EX}')
        with lock:
            os.makedirs(name=os.path.dirname(p=out_file), exist_ok=True)
        package_name: str = f'{TOP_LEVEL_PACKAGE}.{SUBPACKAGE}'
        function_name: str = CONFIG_DATA['plotting_utils'][figure_name]
        logging.debug(
            msg=f'Function {function_name} selected for {figure_name}.')
        module: Any = importlib.import_module(name=package_name)

        logging.debug(msg=f'Attempting to read data from {in_file}.')
        getattr(module, function_name)(in_file=in_file, out_file=out_file,
                                       input_variables=input_params)
        logging.debug(msg=f'Dumped output on {out_file}.')

    # Remove the cached results to force re-computation
    # rmtree(path=NUM_RESULTS_DIR, ignore_errors=True)
    [single_plot_generator(fig, input_variables) for fig in figures_to_plot]


if __name__ == '__main__':
    # Write some unit tests here with simple inputs. The calling code should
    # look as below.
    figures: List[str] = ['throughput_vs_ra_UMa']
    test_output_directory: str = '../test'
    sample_input: Dict[str, List[float]] = {
        'task_size_kb': [4.0],
        'bandwidth_MHz': [4.0],
        'speed_mps': [8.0],
        'hops': [7.0],
        'communication_rad_m': [100.0],
        'number_of_samples_thousands': [1.0]
    }
    rmtree(path=test_output_directory, ignore_errors=True)
    logging.info(msg='Output directory cleared for testing.')
    generate_plots(figures_to_plot=figures,
                   out_dump_directory=test_output_directory,
                   input_variables=sample_input)
    images_created: List[str] = os.listdir(path=test_output_directory)
    logging.info(
        msg=f'Following files found in output directory: {images_created}')
    erase: bool = False  # Set this true to reset the test by removing the
    # images
    if erase:
        rmtree(path=test_output_directory, ignore_errors=True)
