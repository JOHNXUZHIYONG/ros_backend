#!/usr/bin/env python3
# encoding:utf-8

"""
This module implements the worker for output parameter calculation,
generating the pdf and eventual sending by email.

Example:
    To test the module without a WSGI, just use
        $ python3 worker.py

Recommended use: Via flask module
Author: Barman Roy, Swagato
"""
import socket
from hashlib import md5
from pathlib import Path
from shutil import rmtree
from typing import List, Dict

from helper import RESOURCE_DIR, logging, send_email_notification, DL_ROUTE, \
    generate_image_pdf, os, merge_pdfs, REPORT_BASE_NAME, GATEWAY_PORT
from plot_generator import generate_plots

# paths and config variables
SRC_DIR = Path(__file__).parent
OUT_PARENT_DIR = Path(SRC_DIR.parent, 'outbound')
STATIC_PDF: str = "static.pdf"
IMAGE_PDF: str = "image.pdf"


def pipeline(figure_names: List[str],
             email_addresses: List[str],
             plot_parameters: Dict[str, List[float]]) -> None:
    """Process based on the input parameters dumps the report on disk and
    emails a link to download it."""

    # encode batch of emails using first email address
    # Name the output subdirectory after the email address hash
    inner_dir: str = md5(string=email_addresses[0].encode()).hexdigest()
    user_dir: str = os.path.join(OUT_PARENT_DIR, inner_dir)

    # Remove the user directory to erase history
    rmtree(path=user_dir, ignore_errors=True)

    logging.info(msg=f'Attempting to dump contents in {user_dir}.')

    # generate plots for the email address
    generate_plots(figures_to_plot=figure_names,
                   out_dump_directory=user_dir,
                   input_variables=plot_parameters)

    # merge pdf to be sent to user
    image_contents: bytes = generate_image_pdf(image_directory=user_dir)
    static_page: str = os.path.join(RESOURCE_DIR, STATIC_PDF)
    report_contents: bytes = merge_pdfs(static_file=static_page,
                                        image_contents=image_contents)

    logging.debug(msg=f'Successfully merged pdf files.')

    # get report path
    report_full_path: str = os.path.join(user_dir, REPORT_BASE_NAME)
    with open(file=report_full_path, mode='wb') as pointer:
        pointer.write(report_contents)
    logging.debug(
        msg=f'Dumped pdf at {report_full_path}.')

    # get download link
    protocol: str = 'http'
    ip: str = socket.gethostbyname(socket.gethostname())
    link: str = os.path.join(f'{protocol}://{ip}:{GATEWAY_PORT}', DL_ROUTE,
                             inner_dir)

    for email in email_addresses:
        send_email_notification(receiver_address=email, link=link)
        logging.info(msg=f'Sent {link} to {email}.')


if __name__ == '__main__':
    trial_email: str
    # This shebang is to avoid manual editing of email id during testing phase.

    trial_email = 'xu_zhiyong@artc.a-star.edu.sg'

    form_data_list: List[str] = [
        'throughput_vs_ra_UMa']  # CONFIG_DATA['plotting_utils'].keys()  # [

    email_address = [trial_email]

    input_variables = {
        'task_size_kb': [4.0],
        'bandwidth_MHz': [4.0],
        'speed_mps': [8.0],
        'hops': [7.0],
        'communication_rad_m': [100.0],
        'number_of_samples_thousands': [1.0]
    }

    pipeline(form_data_list, email_address, input_variables)
