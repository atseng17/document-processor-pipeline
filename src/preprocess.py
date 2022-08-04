# general functions
import os
import glob
import subprocess
import shutil
import logging
import multiprocessing
from multiprocessing import Pool 
from collections import defaultdict
import pandas as pd
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

# modeling functions
import torch

# preprocessing functions
from pdf2image import convert_from_path
from pdf2image.exceptions import PDFPageCountError
import time


import logging,sys
from opencensus.ext.azure.log_exporter import AzureLogHandler
azure_logging_connection_string=''

logger = logging.getLogger(__name__)

if len(azure_logging_connection_string) > 0:
    logger.addHandler(AzureLogHandler(connection_string=azure_logging_connection_string))
streamHandler = logging.StreamHandler(sys.stdout)

logger.addHandler(streamHandler)

logger.setLevel(logging.INFO)



def get_type_path_dict(root_dir):
    """Get all paths for all datatypes in the dataroom
    Args:
        root_dir (str): path to dataroom

    Returns: 
        doc_type_dict: a dictionary with document type as keys and document paths as value

    """
    doc_type_dict = defaultdict(list)
    for p, d, f in os.walk(root_dir):
        for file in f:
            if file.endswith(tuple([".PDF",".pdf"])):
                doc_type_dict["pdf"].append(os.path.join(p,file))
            elif file.endswith(tuple([".DOCX",".docx"])):
                doc_type_dict["docx"].append(os.path.join(p,file))
            elif file.endswith(tuple([".DOC",".doc"])):
                doc_type_dict["doc"].append(os.path.join(p,file))
            elif file.endswith(tuple([".rtf"])):
                doc_type_dict["rtf"].append(os.path.join(p,file))
            elif file.endswith('.png'):
                doc_type_dict["png"].append(os.path.join(p,file))
            else:
                doc_type_dict[file.split(".")[-1]].append(os.path.join(p,file))
    return doc_type_dict


def doc_to_pdf(doc_path_list,tmp_dir):
    """Libreoffice single conversion from doc(x) and rtf to pdf
    Args:
        doc_path (str): doc(x)/rtf file
        tmp_dir (str): output directory for converted files

    Returns: 
        pdf_path: path to converted doc(x)/rtf file

    """
    pdf_path=[]
    for doc_path in doc_path_list:
        command = ['libreoffice', '--convert-to', 'pdf' , '--outdir', tmp_dir, doc_path]
        subprocess.call(command)
        doc_name =  os.path.basename(doc_path)
        transformed_pdf_name = doc_name.replace(doc_path.split(".")[-1],"pdf")
        transformed_pdf_path = os.path.join(tmp_dir, transformed_pdf_name)
        pdf_path.append(transformed_pdf_path)
    return pdf_path


def doc_to_pdf_multi(doc_dir,out_dir):
    """Libreoffice batch conversion from doc(x) and rtf to pdf
    Args:
        doc_dir (list): list of doc(x)/rtf file paths
        out_dir (str): output directory for converted files

    Returns: 
        None, saves pdfs to out_dir 

    """
    command = ['lowriter', '--headless', '--convert-to', 'pdf', '--outdir', out_dir] + glob.glob(doc_dir)
    subprocess.call(command,shell=False)


def get_failed_pdf(untransformed_path_list, transformed_pdf_file_list):
    """performs a diff on the original doc list and the transformed doc list 
    Args:
        untransformed_path_list (list): list of doc(x)/rtf paths in temporary folder
        transformed_pdf_file_list (list): list of converted doc(x)/rtf paths in temporary folder

    Returns: 
        failed_document_id (list): a list pdf ids keeping track of corrupted docs 

    """
    after_trans = set([os.path.splitext(os.path.basename(doc_to_pdf_path))[0] for doc_to_pdf_path in transformed_pdf_file_list])
    failed_document_id = []
    for document_id, document_path in enumerate(untransformed_path_list):
        if os.path.splitext(os.path.basename(document_path))[0] not in after_trans:
            failed_document_id.append(document_id)
    return failed_document_id

def single_pdf_to_image(pdf_path):
    """Convert a pdf file to a PIL image

    Args:
        pdf_path (str): path to pdf file
    Returns: 
        image (object): PIL image object 
        corrupt (int): "1" indicates a corrupted file
    """
    try:
        image = convert_from_path(pdf_path)[0].convert("RGB")
        corrupt=0
    except PDFPageCountError as e:
        logging.info("Error occured while converting {} to image...{}".format(pdf_path, e))
        image = Image.new('RGB', (100,100))
        corrupt=1
    return image, corrupt


def multi_process_pdf_to_img(pdf_path_list, num_of_processor=None):
    """Multiprocess pdf to PIL conversion

    Args:
        pdf_path (str): path to pdf file
    Returns: 
        image_c_list (list): list of tuples containing (PIL image object, corrupt flag} 
        corrupt (int): An integer where "1" indicates a corrupted file
    """

    if num_of_processor == None:
        proc_num = max(1, multiprocessing.cpu_count()-2)

    else:
        proc_num = min(max(1,num_of_processor), max(1, multiprocessing.cpu_count()-2))

    logging.info(f"using {proc_num} processors for converting pdf to PIL")

    with Pool(proc_num) as p:
        try:
            st_inf = time.time()
            image_c_list = p.map_async(single_pdf_to_image, pdf_path_list).get(300)
            end_inf = time.time()
            logging.info(f"pdf to img took: {end_inf-st_inf}s")
        except multiprocessing.TimeoutError:
            logging.info("Aborting due to timeout")
            logging.info(f"timed out on {pdf_path_list}")
            return None
            
    return image_c_list


