#!/usr/bin/env python3
# encoding:utf-8
"""
This module exposes the workers to the frontend via flask
Author: Barman Roy, Swagato
"""
from multiprocessing import Process
from pathlib import Path
from typing import List

from flask import Flask, render_template, request, send_from_directory
from flask_cors import CORS

from worker import pipeline, os, REPORT_BASE_NAME, OUT_PARENT_DIR, GATEWAY_PORT, \
    logging


# GUI Assets Paths
SRC_DIR = Path(__file__).parent
HTML_TEMPLATES_DIR: Path = Path(SRC_DIR, "templates")

app: Flask = Flask(import_name=__name__, template_folder=HTML_TEMPLATES_DIR)
CORS(app=app)

# declare variables
task_size_kb = 0
bandwidth_MHz = 0
speed_mps = 0
hops = 0
communication_rad_m = 0
num_samples = 0


@app.route('/', methods=['GET'])
def index():
    """webpage for '/'. Default index page

    Returns:
        template for rendering
    """
    return render_template(template_name_or_list='setparameters.html')


@app.route('/setparameters', methods=['GET', 'POST'])
def setParameters():

    return render_template(template_name_or_list="setparameters.html")


@app.route('/setparameters/setusecase', methods=['GET', 'POST'])
def setUseCase():

    global task_size_kb, gateway, bandwidth_MHz, speed_mps, hops, communication_rad_m, num_samples
    print(request.form)

    # get user's input parameters
    user_parameters_dict = request.form

    # assign variables
    task_size_kb = float(user_parameters_dict.get("task_size_range"))
    bandwidth_MHz = float(user_parameters_dict.get("bandwidth_range"))
    speed_mps = float(user_parameters_dict.get("speed_range"))
    hops = float(user_parameters_dict.get("hops_range"))
    communication_rad_m = float(user_parameters_dict.get("comm_rad_range"))
    num_samples = float(user_parameters_dict.get("num_sam_range"))

    return render_template(template_name_or_list='setusecase.html')


@app.route('/answer', methods=['GET', 'POST'])
def answer():
    """webpage for '/answer'. After user has input parameters into form.
    Webpage is able to handle multiple requests from multiple users

    Returns:
        pages
    """

    # Get data input by user, but this does not assume any order. Needs to be refined.
    form_data_dict = request.form

    # clean up data to be sent to pipeline
    form_data_list = list(form_data_dict.keys())
    form_data_list.pop(-1)  # pops email address

    dict_data = form_data_dict.to_dict(flat=False)

    email_address: str = request.form['Email Address']

    # cleanup email address using delimiter ";"
    if len(email_address) != 1:
        email_address = list(email_address.split(","))

    else:
        email_address = list(email_address)

    input_variables = {
        "task_size_kb": [task_size_kb],
        "bandwidth_MHz": [bandwidth_MHz],
        "speed_mps": [speed_mps],
        "hops": [hops],
        "communication_rad_m": [communication_rad_m],
        "number_of_samples_thousands": [num_samples]
    }

    if len(dict_data) != 1:

        # TODO add only when GAP has started
        # pass calculation to a new subprocess
        # report_generator: Process = Process(
        #     target=pipeline, args=(form_data_list, email_address, input_variables))
        # report_generator.start()

        pipeline(form_data_list, email_address, input_variables)

        logging.info(msg=f'Calculation started for {email_address}.')

    else:
        error_message: List[str] = [
            "User did not select any functions to run, " +
            "hence no calculations will be done. Please " +
            "go back to the previous page to declare " +
            "results to be included in the report"]

        dict_data.update({"NOTE": error_message})

    dict_data.update(input_variables.to_dict(flat=False))

    return render_template(template_name_or_list='answer.html',
                           inputs=dict_data)


@app.route(rule='/download_report/<email_hash>', methods=['GET'])
def download_report(email_hash: str):
    """Let the user download the report for his email id."""
    user_dir: str = os.path.join(OUT_PARENT_DIR, email_hash)
    logging.info(msg=f'Download request for {email_hash}')
    return send_from_directory(directory=user_dir, path=REPORT_BASE_NAME,
                               as_attachment=True)


@app.route('/detail', methods=['GET', 'POST'])
def detail():
    """Give the details page."""
    return render_template(template_name_or_list='details.html')


@app.route('/about', methods=['GET', 'POST'])
def details():
    """Give the about page."""
    return render_template(template_name_or_list='about.html')


if __name__ == '__main__':
    host: str = '0.0.0.0'
    app.run(host=host, port=GATEWAY_PORT, debug=True)
