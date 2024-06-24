#!/usr/bin/env python3
# encoding: utf-8

"""Provides some utility functions and configurations to generate the
report."""

import json
import logging
import os
import sys
from email.message import EmailMessage
from glob import glob
from io import BytesIO
from pathlib import Path
from smtplib import SMTP
from typing import Dict, Union

import img2pdf
from PyPDF2 import PdfMerger
from jinja2 import Environment, FileSystemLoader

# paths
SRC_DIR: Path = Path(__file__).parent
CONFIG: Path = Path(SRC_DIR, 'config.json')
RESOURCE_DIR: Path = Path(SRC_DIR.parent, 'resources')
TOP_LEVEL_PACKAGE: str = 'WPA_framework'

SUBPACKAGE: str = 'main'
FRAMEWORK_DIR: Path = Path(SRC_DIR, TOP_LEVEL_PACKAGE)
# Look for all matlab files here.
NUM_RESULTS_DIR: Path = Path(FRAMEWORK_DIR, 'numerical_results')

sys.path.append(FRAMEWORK_DIR.as_posix())

logging.basicConfig(format='%(asctime)s | %(levelname)s: %(message)s',
                    level=logging.DEBUG)

for lib in {'matplotlib', 'img2pdf'}:
    # Silence verbose logs from these libraries
    logging.getLogger(name=lib).setLevel(level=logging.WARNING)

with open(file=CONFIG) as f:
    CONFIG_DATA: Dict[str, Dict] = json.load(fp=f)
    MAIL_CONFIG: Dict[str, Union[int, str]] = CONFIG_DATA.get('mail')
    NIC: str = CONFIG_DATA['network'][
        'GLOB_PROTECT_NIC']  # This is the interface card name for Eric's VMs
    DL_ROUTE: str = CONFIG_DATA['app']['download_route']
    GATEWAY_PORT: int = CONFIG_DATA['app'][
        'port']  # noqa: F841 use this variable in other modules

# Mail credentials
USER: str = MAIL_CONFIG.get("USER")
DOMAIN: str = MAIL_CONFIG.get(
    "EMAIL_DOMAIN")  # In-house SMTP service does not accept domain as part
# of credential
PASSWORD: str = MAIL_CONFIG.get("PASSWORD")
HOST: str = MAIL_CONFIG.get("HOST")
MAIL_SERVER_PORT: int = MAIL_CONFIG.get("PORT")

# Mail body
BODY_FILE: str = MAIL_CONFIG.get("BODY_FILE")
SUBJECT: str = MAIL_CONFIG.get("SUBJECT")
REPORT_BASE_NAME: str = MAIL_CONFIG.get("REPORT_BASE_NAME")
CONTENT_TYPE: str = MAIL_CONFIG.get("CONTENT_TYPE")

# File extensions
INPUT_EX: str = CONFIG_DATA["file_types"]["input_data"]
OUTPUT_EX: str = CONFIG_DATA["file_types"]["output_data"]


def send_email_notification(receiver_address: str,
                            link: str) -> None:
    """
    Send the email with a link to download the report.
    """
    # Create the container email message.
    # declares email body, subject, recipient, and sender
    msg = EmailMessage()
    msg['Subject'] = SUBJECT
    sender: str = f'{USER}@{DOMAIN}'
    msg['From'] = sender
    msg['To'] = receiver_address

    # getting mail body and rendering the link to be sent
    mail_body: str = Environment(
        loader=FileSystemLoader(searchpath=RESOURCE_DIR)) \
        .get_template(name=BODY_FILE) \
        .render(link=link)

    # set content instead of attaching it as an attachment
    msg.set_content(mail_body, subtype=CONTENT_TYPE)
    logging.debug(msg=f'Rendered email template from {BODY_FILE}.')

    # Send the email via our own SMTP server.
    # TODO on SMG Dev, starttls causes an smtplib.SMTPNotSupportedError
    # TODO on SMG Dev, login causes an smtplib.SMTPAuthenticationError
    with SMTP(host=HOST, port=MAIL_SERVER_PORT) as session:
        session.set_debuglevel(debuglevel=1)
        session.starttls()  # Server refuses connection without TLS
        session.login(user=USER, password=PASSWORD)
        session.send_message(msg=msg,
                             from_addr=sender,
                             to_addrs=receiver_address)
    logging.info(msg=f'Email sent to {receiver_address}.')


def generate_image_pdf(image_directory: str, extension: str = 'jpg') -> bytes:
    """
    Create a new pdf file at out_file_path based on the images in the
    image_directory.

    The directory of out_file_path must already exist
    """
    return img2pdf.convert(
        glob(pathname=os.path.join(image_directory, f'*.{extension}')))


def merge_pdfs(static_file: str, image_contents: bytes) -> bytes:
    """
    This function concatenates the two pdfs into a single in-memory byte array.
    Args:
        static_file: Disk path of the static pdf containing the glossary of
        terms. Must be readable.
        image_contents: The byte array of pdfs in memory generate by img2pdf
    Returns: The byte-array corresponding to the merged pdf.
    """
    with PdfMerger() as merged:
        with BytesIO(initial_bytes=image_contents) as dynamic:
            for pdf in (static_file, dynamic):
                # PdfMerger.append accepts disk file path or byte buffer
                merged.append(fileobj=pdf)
        # PdfMerger does not expose the underlying byte-array directly.
        # Hence, write to another buffer
        with BytesIO() as final_buffer:
            merged.write(fileobj=final_buffer)
            logging.debug(msg=f'Merged {static_file} with image data.')
            return final_buffer.getvalue()  # The final report content
