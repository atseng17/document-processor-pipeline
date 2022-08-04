# general functions
import os
import shutil
import datetime


def make_tmp_dir(tmp_file_list, dest_dir):
    """Make a number of temporary folder for storing doc/docx/rtf files for further transformations

    Args:
        tmp_file_list (list): list of dic.docx/rtf paths
        dest_dir (str): path to the temporary folder

    Returns: 
        pdf_embeddings (numpy.ndarray): an array of nparrays of embeddings of images 
        blank_check (numpy.ndarray): an array of 1/0 where 1 indicates corrupted image 
    """
    new_path_list = []
    os.makedirs(dest_dir,  exist_ok=True)
    fid=-1
    for i, source_file in enumerate(tmp_file_list):
        if i%200==0:
            fid+=1
            os.makedirs(os.path.join(dest_dir, f"tmp_{fid}"),  exist_ok=True)
        filename = os.path.splitext(os.path.basename(source_file))[0]
        extension = os.path.splitext(os.path.basename(source_file))[-1].lower()
        # format current timestamp and append it to filename
        timestamp = str(datetime.datetime.now()).replace(" ", "_")
        timestamp = timestamp.replace(":","_")
        timestamp = timestamp.replace(".","_")
        new_doc_name = timestamp + filename + extension
        destination = os.path.join(dest_dir, f"tmp_{fid}",new_doc_name)
        shutil.copy(source_file, destination)
        new_path_list.append(destination)

    return new_path_list


    