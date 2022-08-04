# general functions
import os
import sys
import glob
import threading
import datetime
import uuid
import pickle
import time
import logging
import yaml
import argparse
import shutil
import psutil
import traceback
import signal
import pandas as pd
import numpy as np
import multiprocessing
from multiprocessing import Pool 
from cmath import log
from fastapi import FastAPI,Request
from ast import Try, literal_eval
from collections import defaultdict

# modeling functions
import torch
from pdf2image import convert_from_path

from transformers import (LayoutLMv2FeatureExtractor, 
    LayoutLMv2Tokenizer, 
    LayoutLMv2Processor, 
    LayoutLMv2ForSequenceClassification
)

from transformers import logging as transformer_logger
transformer_logger.set_verbosity_error()


from src.model import (assign_discriptions,
                   batch_inference,
                   cluster_kmeans_elbow,
                   pdf2keywords_keybert,    
                   process_png            
)

from src.preprocess import (doc_to_pdf_multi,
                   get_type_path_dict,
                   get_failed_pdf,
                   multi_process_pdf_to_img               
)

from src.utils import make_tmp_dir



from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient, __version__


import logging
from opencensus.ext.azure.log_exporter import AzureLogHandler
azure_logging_connection_string=''

logger = logging.getLogger(__name__)

if len(azure_logging_connection_string) > 0:
    logger.addHandler(AzureLogHandler(connection_string=azure_logging_connection_string))
streamHandler = logging.StreamHandler(sys.stdout)

logger.addHandler(streamHandler)

logger.setLevel(logging.INFO)

app = FastAPI()

@app.post("/")
async def handle_post(postRequest: Request):
    # default values
    dataroom_uri = ""
    output_path = ""
    output_file_name = ""
    num_clusters = 0    
    
    # get request
    req_info = await postRequest.json()
    input_request = dict(req_info)
    # check all required keys are there and set defaults for optional ones
    if "input_path" not in input_request.keys():
        return {"Error":"no input path defined"}

    for key,value in zip(input_request.keys(),input_request.values()):
        if key == "input_path":
            dataroom_uri = value
        if key == "output_path":
            output_path = value
            output_file_name = value.split('/')[-1]
        if key == "num_clusters":
            num_clusters = value

    # if output file name isnt provided, create one
    if output_file_name == "":
        date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_file_name = date + "_results.json"
    
    # if output not given, its same as input
    if output_path == "":
        output_path = dataroom_uri
    
    logger.info("uri "+str(dataroom_uri))
    logger.info("output_uri " + str(output_path))
    logger.info("output file " + str(output_file_name))
    logger.info("num clusters " + str(num_clusters))
    
    # get the azure connection string from model config yaml
    # and determine which connection string to use / if one can be used for this data room
    azure_connection_string = ""
    # get the data room string from the URI
    data_room = dataroom_uri.split('.')[0][len("https://"):]
    # read from config
    CONFIGPATH = "config.yaml"
    with open(CONFIGPATH, "r") as f:
        config = yaml.load(f,Loader=yaml.Loader)
        # check all strings in the config file to see if one matches current data room
        for connection_string in config["connection_strings"]:
            # get the data room per each string
            data_room_for_connection_string = connection_string.split("=")[2][:-len(";AccountKey")]
            # check if the data room matches the one for that string
            if data_room == data_room_for_connection_string:
                # if so, set string and exit loop
                azure_connection_string = connection_string
                break
    
    # if doesnt have access to that data room, then return an error
    if azure_connection_string == "":
        return {"error":"API does not have access to data room " + data_room}
    
    
    # get an estimate from the data room
    output = generate_estimate(dataroom_uri,azure_connection_string)
    output["output_file_name"] = output_file_name
    output["output_graph_name"] = output_file_name.replace("json","png")
    
    # start processing in background before returning estimates to user
    thread_name = "dataroom_processor"+datetime.datetime.now().strftime("%m-%d-%Y-%H-%M-%S")+"est" + str(output["time_estimate_hours"])
    processing_thread = threading.Thread(target=do_dataroom, name=thread_name, args=(dataroom_uri,output_path,output_file_name,num_clusters,azure_connection_string))
    processing_thread.daemon = True
    processing_thread.start()
    
    return output
    
@app.get("/")
async def root():
    """returns the started time, and the time left for all data processor threads

    Returns:
        dict: list of all running processor thread start times and time remaining
    """
    output = dict()
    output["running_data_processors"] = []
    # list running threads    
    for thread in threading.enumerate(): 
        if "dataroom_processor" in thread.name:
            dataroom_processor = dict()
            # take the time the thread started at and its estimate from its name extrapolate hours left
            started_at_str = thread.name[len("dataroom_processor"):thread.name.find("est")]
            started_at = datetime.datetime.strptime(started_at_str, "%m-%d-%Y-%H-%M-%S")
            dataroom_processor["started at"] = started_at.strftime("%Y/%m/%d %H:%M:%S")
            difference = datetime.datetime.now() - started_at
            seconds_since = difference.total_seconds()
            seconds_left = float(thread.name[thread.name.find("est") + 3:])*3600 - seconds_since
            hours_left = round(seconds_left/3600, 2)
            dataroom_processor["hours left"] = hours_left
            output["running_data_processors"].append(dataroom_processor)
    # get cpu and memory info
    output["cpu_usage_percent"] = psutil.cpu_percent(1)
    output["memory_usage_percent"] = psutil.virtual_memory()[2]
    
    return output


def generate_estimate(dataroom_uri,azure_connection_string):
    """Returns an estimate dict for size and processing time

    Args:
        dataroom_uri (string): URL of the dataroom as an input
        azure_connection_string (string): connection string to access azure blob storage

    Returns:
        dict: size and processing time estimates in mb and seconds
    """
    # create blob storage client    
    blob_service_client = BlobServiceClient.from_connection_string(azure_connection_string)
    
    # get just the name of the data room
    logger.info("splitting old way " + str(dataroom_uri))
    dataroom_name_full = dataroom_uri.split(".net")[-1][1:]
    dataroom_name = dataroom_name_full.split("/")[0]
    if dataroom_name[-1] == "/":
        dataroom_name = dataroom_name[:-1]
    
    # folder to filter on, will be [] if its the full dataroom as input
    dataroom_folder = dataroom_name_full.split("/")[1:]
    if dataroom_folder[-1] == '':
        dataroom_folder = dataroom_folder[:-1]
    dataroom_folder_filter = "/".join(dataroom_folder)
    # create a container client from the blob client and the data room name
    container_client = blob_service_client.get_container_client(dataroom_name)
    
    
    # base estimate is 0.0
    estimate = dict()
    estimate["time_estimate_hours"] = 0.0
    estimate["size_estimate_kb"] = 0.0
    
    # List the blobs in the container
    blob_list = container_client.list_blobs()
    for blob in blob_list:
        # if their is either no filter, or the blob matches the filter
        # to pass the filter, the folder path must be a substring of the name, and the character after it must be /
        if (len(dataroom_folder)==0) or ((dataroom_folder_filter in blob.name) and (blob.name[len(dataroom_folder_filter)]=='/')):
            # each file adds 5 seconds of processing time
            estimate["time_estimate_hours"] += 5.0
            # each file adds ~ 0.25kb of output size to the output json
            estimate["size_estimate_kb"] += 0.22
    # convert to hours
    estimate["time_estimate_hours"] = estimate["time_estimate_hours"]/3600
    
    return estimate

def do_dataroom(dataroom_uri,output_path,output_file_name,num_clusters,azure_connection_string):
    # try to process the dataroom
    try:
        logger.info("starting do dataroom")
        dataroom_path = download_dataroom(dataroom_uri,azure_connection_string)
        logger.info("dataroom is downloaded")
        results_file_location = cluster_dataroom(dataroom_path,output_file_name,num_clusters)
        logger.info("data room clustering is done")
        output_results_to_dataroom(dataroom_uri,output_path,results_file_location,azure_connection_string)
        logger.info("output sent to dataroom")
        clean_up_files(dataroom_path)
        logger.info("processing complete, results in place")
        
    except Exception as e:
        # any exception gets output to the the output location
        # log exception
        logger.info("There was an exception thrown while processing")
        logger.exception(e)

        # write exception output file
        exception_file = os.path.join("/tmp/",output_file_name)
        with open(exception_file,"w") as e_file:
            exception_name = str(e)
            exception_traceback = str(traceback.format_tb(e.__traceback__))            
            exception_json = {"exception thrown":exception_name,"traceback":exception_traceback}
            e_file.write(str(exception_json))
        logger.info("Exception will be output as results")
        # output as result
        output_results_to_dataroom(dataroom_uri,output_path,exception_file,azure_connection_string)
        # clean up the exception file
        os.remove(exception_file) 
    
    
    

def clean_up_files(dataroom_path):
    # delete the path that the data room is downloaded to
    if os.path.exists(dataroom_path):
        shutil.rmtree(dataroom_path)
    

def download_dataroom(dataroom_uri,azure_connection_string):
    # download to uuid folder
    folder_name = "/tmp/" + str(uuid.uuid4())
    os.makedirs(folder_name)
    # create a blob client
    blob_service_client = BlobServiceClient.from_connection_string(azure_connection_string)
    
    
    # get just the name of the data room
    dataroom_name_full = dataroom_uri.split(".net")[-1][1:]
    dataroom_name = dataroom_name_full.split("/")[0]
    if dataroom_name[-1] == "/":
        dataroom_name = dataroom_name[:-1]
    
    # folder to filter on, will be [] if its the full dataroom as input
    dataroom_folder = dataroom_name_full.split("/")[1:]
    if dataroom_folder[-1] == '':
        dataroom_folder = dataroom_folder[:-1]
    dataroom_folder_filter = "/".join(dataroom_folder)
    # create a container client from the blob client and the data room name
    container_client = blob_service_client.get_container_client(dataroom_name)

    # List the blobs in the container
    blob_list = container_client.list_blobs()
    for blob in blob_list:
        logger.info(blob.name)
        # if their is either no filter, or the blob matches the filter
        # to pass the filter, the folder path must be a substring of the name, and the character after it must be /
        if (len(dataroom_folder)==0) or ((dataroom_folder_filter in blob.name) and (blob.name[len(dataroom_folder_filter)]=='/')):
            # get the local path to where the blob will be downloaded
            blob_path = os.path.join(folder_name,blob.name)
            # if blob has a / in the name, it is / is in a folder
            if '/' in blob.name:
                # get the folder name for the blob
                folder_inside_container = os.path.dirname(blob.name)
                # combine the folder in the container with the local path where the blob will be downloaded
                local_folder_to_make = os.path.join(folder_name,folder_inside_container)
                # if theres a blob sharing name with a folder this will crash
                os.makedirs(local_folder_to_make,exist_ok=True)
            
            # download the blob to its path, not that any prerequisite folders are created
            with open(blob_path,"wb") as local_blob_file:
                local_blob_file.write(container_client.download_blob(blob.name).readall())
        
    return folder_name


def cluster_dataroom(dataroom_path,output_file_name,num_clusters):
    """Perform clustering on a given dataroom
    Args: 
        dataroom_path (str): path to downloaded dataroom
        output_file_name (str): path to json output
        num_clusters (int): user defined number of clusters, if not 0, will override elbow method

    Returns: 
        OUTPUT_PATH (str): path to json output
    """
    pd.set_option('max_colwidth', -1)
    np.random.seed(3)
    TIMESTR = time.strftime("%Y%m%d-%H%M%S")
    CONFIGPATH = "config.yaml"
    LOGPATH = f"{TIMESTR}.log"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # open config file and load
    with open(CONFIGPATH, "r") as f:
        config = yaml.load(f,Loader=yaml.Loader)

    # load config info
    DATAROOM_ROOT_PATH = dataroom_path 
    
    DOC_TMP_FOLDER = "/tmp/" + str(uuid.uuid4())
    TRANSFROMED_PDF_FOLDER = DOC_TMP_FOLDER + "/pdfs" 
    TOP_N_KEYWORDS = config['keyword_extraction']['top_n_keywords']
    KEYPHRASE_NGRAM_RANGE = config['keyword_extraction']['keyphrase_ngram_range']
    ENCODING_LIMIT = config['layoutlm_model']['encoding_limit']
    BATCH_SIZE = config['layoutlm_model']['batch_size']
    # changable user K
    FIXK = num_clusters
    MAX_TRIES = config['kmeans_model']['max_tries']
    # This is based on the P3v2 instance type having 4 cores
    NUM_PROCESSOR = 4 
    OUTPUT_PATH = dataroom_path + "/"+output_file_name 

    # remove intermediate folders from previous runs
    if os.path.exists(DOC_TMP_FOLDER):
        shutil.rmtree(DOC_TMP_FOLDER)

    if os.path.exists(TRANSFROMED_PDF_FOLDER):
        shutil.rmtree(TRANSFROMED_PDF_FOLDER)

    os.makedirs(TRANSFROMED_PDF_FOLDER)
    logging.basicConfig(filename=LOGPATH,level=logging.INFO,format= '[%(asctime)s] %(levelname)s - %(message)s',datefmt='%H:%M:%S')
    
    # Process Doc(x) and RTF Files
    # Get document type information
    logger.info("processing docx and rtf")
    doc_type_dict = get_type_path_dict(DATAROOM_ROOT_PATH)
    existing_doc_types = list(doc_type_dict.keys())

    doc_docx_rtf_path_list=[]
    for doc_type in ['doc','docx','rtf']:
        if doc_type in existing_doc_types:
            doc_docx_rtf_path_list.extend(doc_type_dict[f"{doc_type}"])

    pdf_path_list=[]
    for doc_type in ['pdf']:
        if doc_type in existing_doc_types:
            pdf_path_list.extend(doc_type_dict[f"{doc_type}"])

    png_path_list=[]
    for doc_type in ['png']:
        if doc_type in existing_doc_types:
            png_path_list.extend(doc_type_dict[f"{doc_type}"])

    # Process doc(x)/rtf files in dataroom
    if not len(doc_docx_rtf_path_list)==0:

        st = time.time()
        # Save docs, doc, rtf to seperate temporary folders and keep track of them
        new_doc_path_list = make_tmp_dir(doc_docx_rtf_path_list, DOC_TMP_FOLDER)

        # Libreoffice has a limit of 200 documents when multiprocessing, thus process each folder with 200 one by one
        for subfolder in glob.glob(os.path.join(DOC_TMP_FOLDER,"*/")):
            if not len(glob.glob(os.path.join(subfolder,"*.doc")))==0:
                doc_tmp_folder = os.path.join(subfolder,"*.doc")
                doc_to_pdf_multi(doc_tmp_folder,TRANSFROMED_PDF_FOLDER)
            if not len(glob.glob(os.path.join(subfolder,"*.docx")))==0:
                doc_tmp_folder = os.path.join(subfolder,"*.docx")
                doc_to_pdf_multi(doc_tmp_folder,TRANSFROMED_PDF_FOLDER)
            if not len(glob.glob(os.path.join(subfolder,"*.rtf")))==0:
                doc_tmp_folder = os.path.join(subfolder,"*.rtf")
                doc_to_pdf_multi(doc_tmp_folder,TRANSFROMED_PDF_FOLDER)
        
        end = time.time()
        logging.info(f"doc(x)/rtf to pdf format, spent: {end-st}s")

        # Get transformed pdf path list for further processing
        transformed_pdf_file_list = glob.glob(os.path.join(TRANSFROMED_PDF_FOLDER , f"**/*.pdf"), recursive=True)

        # Format dataframe for storing results for doc, docx, rtf
        final_output = pd.DataFrame(doc_docx_rtf_path_list, columns =['document_path'], dtype = str)
        final_output["corrupt"]=0

        # Keep track of corrupted docs
        failed_document_id = get_failed_pdf(new_doc_path_list, transformed_pdf_file_list)
        final_output.iloc[failed_document_id, final_output.columns.get_loc('corrupt')] = 1

        tmp_doc_pdf_list = []
        pdf_id=0
        for i in range(len(doc_docx_rtf_path_list)):
            if i in failed_document_id:
                tmp_doc_pdf_list.append("None")
            else:
                tmp_doc_pdf_list.append(transformed_pdf_file_list[pdf_id])
                pdf_id+=1

        final_output["pdf_path"]=tmp_doc_pdf_list

    else:
        # Create empty dataframe
        final_output= pd.DataFrame([], columns =['document_path','pdf_path', 'corrupt'], dtype = str)


    if not len(pdf_path_list)==0:
        # Append original pdf list to existing dataframe
        pdf_output = pd.DataFrame(pdf_path_list, columns =['document_path'], dtype = str)
        pdf_output['corrupt']=0
        pdf_output["pdf_path"]=pdf_path_list
        final_output = pd.concat([final_output, pdf_output], axis=0).reset_index(drop=True)

        df_uncorrupted_documents = final_output[final_output['corrupt']==0]
        df_corrupted_documents = final_output[final_output['corrupt']==1]
        final_output = pd.concat([df_uncorrupted_documents, df_corrupted_documents], axis=0).reset_index(drop=True)
        logger.info("made df_uncorrupted_documents i")

    else:
        # Create empty dataframe
        df_uncorrupted_documents = final_output[final_output['corrupt']==0]
        df_corrupted_documents = final_output[final_output['corrupt']==1]
        final_output = pd.concat([df_uncorrupted_documents, df_corrupted_documents], axis=0).reset_index(drop=True)
        logger.info("made df_uncorrupted_documents e")
        
    # Process PDF Files
    # Process all non corrupted pdfs
    if not len(final_output[final_output['corrupt']==0])==0:
        st = time.time()
        pdf_embeddings=[]
        blank_check=[]
        corrupted_pdf_id_list=[]
        remaining_pdf_list = final_output[final_output["corrupt"]==0]["pdf_path"].tolist()
        pdf_to_pil_batch_sz = BATCH_SIZE
        total_batches = int(np.ceil(len(remaining_pdf_list) / pdf_to_pil_batch_sz))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for i in range(total_batches):
            # pdf to pil format
            pdf_file_list = remaining_pdf_list[i*pdf_to_pil_batch_sz:(i+1)*pdf_to_pil_batch_sz]
            logger.info(f"multi process pdf started for batch {i+1} / {total_batches}")
            image_c_list = multi_process_pdf_to_img(pdf_file_list,NUM_PROCESSOR)
            logger.info(f"multi process pdf finished for batch {i+1} / {total_batches}")
            
            # skip batch if PDFs failed to convert
            skip_batch = (image_c_list is None)
            
            if not skip_batch:
                corrupted_pdf_ids = [c for pil_image,c in image_c_list]
                pil_image_list = [pil_image for pil_image,c in image_c_list]

                # Document embedding creation via LayoutLM model batch inference, if inference takes to long, skip batch
                logger.info(f"batch inference about to start on device {device} {i+1} / {total_batches}")
                with Pool(1) as p:
                    try:
                        st_inf = time.time()
                        sub_pdf_embeddings, sub_blank_check = p.apply_async(batch_inference, args=(pil_image_list, BATCH_SIZE, device, int(ENCODING_LIMIT))).get(600)
                        end_inf = time.time()
                        logging.info(f"Batch Inference, spent: {end_inf-st_inf}s")
                    except multiprocessing.TimeoutError:
                        logging.info("Aborting due to timeout")
                        logging.info(f"timed out batch {pdf_file_list}")
                        skip_batch = True
                        # also skip batch if timeout failed
                
            if skip_batch:
                # set to blank if skipping
                logger.info(f"Batch {pdf_file_list} is skipped due to timeout in pdf or inference")
                sub_pdf_embeddings = np.zeros([len(pdf_file_list),768])
                sub_blank_check = np.ones(len(pdf_file_list))*int(102)
            
            logger.info(f"batch inference done {i+1}/ {total_batches}")
            pdf_embeddings.append(sub_pdf_embeddings)
            corrupted_pdf_id_list.extend(corrupted_pdf_ids)
            blank_check.extend(list(sub_blank_check))

            if not skip_batch:
                del image_c_list
                del pil_image_list

            
        # Keep track of corrupted pdfs
        blank_pdf_check_list = [int(bc==102) for bc in blank_check]
        corrupted_pdf_list = corrupted_pdf_id_list
        corrupted_blank_pdf_list=[a or b for a,b in zip(corrupted_pdf_id_list,blank_pdf_check_list)]

        end = time.time()
        logging.info(f"pdf to PIL format, spent: {end-st}s")
        
    else:
        # if no pdf files in dataroom, create empty list as placeholder
        corrupted_pdf_list = []
        blank_pdf_check_list = []
        corrupted_blank_pdf_list = []
        pdf_embeddings = []
        
    # process png files
    if not len(png_path_list)==0:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        png_embeddings, blank_check_png_list = process_png(png_path_list,  BATCH_SIZE, device, int(ENCODING_LIMIT))
        # Append png information to dataframe
        png_output = pd.DataFrame(png_path_list, columns =['document_path'], dtype = str)
        png_output["corrupt"]=0
        png_output["pdf_path"]="image_file"
        final_output = pd.concat([final_output, png_output], axis=0).reset_index(drop=True)

    else:
        # if no png files in dataroom, create empty list as placeholder
        blank_check_png_list=[]
        png_embeddings = []

    # Update final output with pdf and png information
    number_of_corrupted_doc_xdocx_rtf = len(final_output) - len(corrupted_pdf_list) - len(blank_check_png_list)
    final_output["corrupted_pdf"] = corrupted_pdf_list + [1]*(number_of_corrupted_doc_xdocx_rtf) + blank_check_png_list
    final_output["no_content"] = blank_pdf_check_list + [1]*(number_of_corrupted_doc_xdocx_rtf) + blank_check_png_list

    df_blank_documents = final_output[(final_output["corrupted_pdf"]==0) & (final_output["no_content"]==1)]
    df_corrupted_documents = final_output[(final_output["corrupted_pdf"]==1)]
    df_uncorrupted_not_blank =  final_output[(final_output["no_content"]==0) & (final_output["no_content"]==0)]
    final_output = pd.concat([df_uncorrupted_not_blank,df_blank_documents, df_corrupted_documents], axis=0)
    final_output = final_output.reset_index(drop=True)
    logger.info("reset index")
    # Document Clustering
    # do clustering
    if not len(df_uncorrupted_not_blank) == 0:

        st = time.time()
        # Format embeddgings to dataframe
        document_embeddings = np.expand_dims(np.vstack(pdf_embeddings+png_embeddings), axis=1).tolist()
        document_embeddings_df = pd.DataFrame(document_embeddings, columns=["token0v2"])
        document_embeddings_df["corrupt"] = corrupted_blank_pdf_list + blank_check_png_list
        un_corrupted_embeddings = document_embeddings_df[document_embeddings_df["corrupt"]==0].reset_index(drop=True)

        # Kmeans with Elbow method
        output_plot_path = OUTPUT_PATH.replace("json","png")
        cluster_label_list, confidence_score_list, closest_to_centroid = cluster_kmeans_elbow(un_corrupted_embeddings, user_define_k = FIXK, max_tries = MAX_TRIES, show_tsne=True,plot_location=output_plot_path)
        end = time.time()
        logging.info(f"Clustering, spent: {end-st}s")


        # update cluster ID and confidence score to final output
        cluster_label_col = cluster_label_list + ["no_content"]*len(df_blank_documents) + ["corrupt"]*len(df_corrupted_documents)

        confidence_score_col = confidence_score_list + [0]*(len(final_output)-len(cluster_label_list))
        final_output["cluster_label"] = cluster_label_col
        final_output["confidence_score"] = confidence_score_col

    else:
        final_output["cluster_label"] = "corrupt"
        final_output["confidence_score"] = 0


    # Document Description Generation
    if not len(df_uncorrupted_not_blank) == 0:
        st = time.time()
        # Load OCR
        feature_extractor = LayoutLMv2FeatureExtractor()
        tokenizer = LayoutLMv2Tokenizer.from_pretrained("microsoft/layoutlmv2-base-uncased")
        processor = LayoutLMv2Processor(feature_extractor, tokenizer)

        # Keyword generation
        centroid_discription_paths = final_output.iloc[closest_to_centroid]["pdf_path"]
        centroid_discriptions  = centroid_discription_paths.apply(pdf2keywords_keybert, 
                                                                processor=processor, 
                                                                keyphrase_ngram_range = literal_eval(KEYPHRASE_NGRAM_RANGE), 
                                                                top_n = TOP_N_KEYWORDS)
        end = time.time()
        logging.info(f"Keyword extraction, spent: {end-st}s")

        # Add label discription to final output
        final_output['label_description'] = final_output.apply(lambda x: assign_discriptions(x['cluster_label'],centroid_discriptions), axis=1)
        final_output.drop(columns=['corrupt', 'no_content', 'pdf_path', 'corrupted_pdf'], axis = 1, inplace = True)

    else:
        # If all doc(x)/rtf/png/pdf are corrupted or no doc(x)/rtf/png/pdf
        # in the dataroom, then generate an empty dataframe output
        final_output['label_description'] = "corrupt"
        final_output.drop(columns=['corrupt', 'no_content', 'pdf_path', 'corrupted_pdf'], axis = 1, inplace = True)



    # Final JSON Output
    # Merge informaton of other data types into final output
    other_type_path_list = []

    for doc_type, doc_path_list in doc_type_dict.items():
        if doc_type not in ["pdf","doc","docx","rtf","png"]:
            for doc_path in doc_path_list:
                other_type_path_list.append([doc_path, doc_type ,"1", doc_type])

    other_type_df = pd.DataFrame(other_type_path_list, columns=['document_path', 'cluster_label', 'confidence_score', 'label_description'])
    final_output = pd.concat([final_output, other_type_df], axis=0).reset_index(drop=True)
    
    final_output['document_path'] = final_output['document_path'].apply(lambda x:x[len(DOC_TMP_FOLDER):])

    final_output.to_json(OUTPUT_PATH, orient='records', lines=True)

    # Remove intermediate files
    if os.path.exists(DOC_TMP_FOLDER):
        shutil.rmtree(DOC_TMP_FOLDER)

    if os.path.exists(TRANSFROMED_PDF_FOLDER):
        shutil.rmtree(TRANSFROMED_PDF_FOLDER)
    
    logger.info("done " +str(OUTPUT_PATH))
    return OUTPUT_PATH

def output_results_to_dataroom(dataroom_uri,output_path,results_file_location,azure_connection_string):
    
    # connection, same as estimate
    blob_service_client = BlobServiceClient.from_connection_string(azure_connection_string)
    
    # same as estimate
    dataroom_name = dataroom_uri.split(".net")[-1][1:]
    if dataroom_name[-1] == "/":
        dataroom_name = dataroom_name[:-1]
    
    # get the final name of the results file
    results_file_name = results_file_location.split("/")[-1]
    
    # create a new blob client for this non-existant blob
    blob_client = blob_service_client.get_blob_client(container=dataroom_name, blob=results_file_name)
    logger.info("\nUploading to Azure Storage as blob:\n\t" + results_file_name)
    # Upload the created file
    with open(results_file_location, "rb") as data:
        blob_client.upload_blob(data)
    
    # same for latest but different name and overwrite=true
    blob_client = blob_service_client.get_blob_client(container=dataroom_name, blob="latest_results.json")
    logger.info("\nUploading to Azure Storage as blob:\n\t" + results_file_name)
    # Upload the created file
    with open(results_file_location, "rb") as data:
        blob_client.upload_blob(data=data,overwrite=True)
    
    # same thing, but replace json with png in the results file, to get the plot
    # only if it exists
    plot_file_name = results_file_name.replace("json","png")
    plot_file_location = results_file_location.replace("json","png")
    if os.path.exists(plot_file_location):
        # create a new blob client for this non-existant blob
        blob_client = blob_service_client.get_blob_client(container=dataroom_name, blob=plot_file_name)
        logger.info("\nUploading to Azure Storage as blob:\n\t" + plot_file_name)
        # Upload the created file
        with open(plot_file_location.replace("json","png"), "rb") as data:
            blob_client.upload_blob(data)
    
    return
