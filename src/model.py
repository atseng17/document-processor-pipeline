# general functions
import logging
import sys
import random
from collections import Counter
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from opencensus.ext.azure.log_exporter import AzureLogHandler

# modeling functions
import torch
from keybert import KeyBERT
from pdf2image import convert_from_path
from transformers import (LayoutLMv2FeatureExtractor, 
    LayoutLMv2Tokenizer, 
    LayoutLMv2Processor, 
    LayoutLMv2ForSequenceClassification
)
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist

# metric functions
from sklearn.metrics import (
    pairwise_distances_argmin_min, 
    silhouette_score, 
    silhouette_samples
)
from scipy.spatial.distance import cdist

# preprocess functions
from src.preprocess import single_pdf_to_image


azure_logging_connection_string=''
logger = logging.getLogger(__name__)
if len(azure_logging_connection_string) > 0:
    logger.addHandler(AzureLogHandler(connection_string=azure_logging_connection_string))
streamHandler = logging.StreamHandler(sys.stdout)
logger.addHandler(streamHandler)
logger.setLevel(logging.INFO)

RANDOM_STATE = 0

def trim_encoded_inputs(encoded_inputs, limit):
    """Trim encoded sentences to a fixed length

    Args: 
        encoded_inputs (object): LayoutLMv2Processor with document encoding information
        limit (int): limitation of encoding length
    Returns: 
        encoded_inputs (object): LayoutLMv2Processor with updated encoding information
    """
    if encoded_inputs["input_ids"].shape[1] > limit:
        encoded_inputs["input_ids"] = encoded_inputs["input_ids"][:, :limit]
        encoded_inputs["bbox"] = encoded_inputs["bbox"][:, :limit, :]
        encoded_inputs["token_type_ids"] = encoded_inputs["token_type_ids"][:, :limit]
        encoded_inputs["attention_mask"] = encoded_inputs["attention_mask"][:, :limit]
    return encoded_inputs


def single_inference(pdf_path, device, encoding_limit):
    """LayoutLM model single document inference

    Args: 
        pdf_path (str): path to pdf file
        device (torch.device): the device on which a torch.Tensor is or will be allocated
        encoding_limit (int): limitation of encoding length
    Returns: 
        pdf_embeddings (numpy.ndarray): embeddings of single pdf 
    """
    feature_extractor = LayoutLMv2FeatureExtractor()
    tokenizer = LayoutLMv2Tokenizer.from_pretrained("microsoft/layoutlmv2-base-uncased")
    processor = LayoutLMv2Processor(feature_extractor, tokenizer)
    image = convert_from_path(pdf_path)[0]
    encoded_inputs = trim_encoded_inputs(processor(image, return_tensors="pt"), limit=encoding_limit)
    for k,v in encoded_inputs.items():
        encoded_inputs[k] = v.to(device)
    model = LayoutLMv2ForSequenceClassification.from_pretrained("microsoft/layoutlmv2-base-uncased", 
                                                                output_hidden_states = True)
    model.to(device)
    outputs = model(**encoded_inputs)
    pdf_embeddings = outputs.hidden_states[-1][0][0].cpu().detach().numpy()
    return pdf_embeddings


def batch_inference(img_list, batch_sz, device, encoding_limit):
    """Batch inference code for processing PIL objects

    Args:
        img_list (list): list of PIL objects
        device (torch.device): the device on which a torch.Tensor is or will be allocated
        encoding_limit (int): maximum sequence length

    Returns: 
        pdf_embeddings (numpy.ndarray): an array of numpy.ndarrays of embeddings of images 
        blank_check (numpy.ndarray): an array of 1/0 where 1 indicates corrupted image 
    """
    feature_extractor = LayoutLMv2FeatureExtractor()
    logger.info("fe")
    tokenizer = LayoutLMv2Tokenizer.from_pretrained("microsoft/layoutlmv2-base-uncased")
    logger.info("token")
    processor = LayoutLMv2Processor(feature_extractor, tokenizer)
    logger.info("processor")
    model = LayoutLMv2ForSequenceClassification.from_pretrained("microsoft/layoutlmv2-base-uncased", 
                                                            output_hidden_states = True)
    logger.info("model")
    model.to(device)
    logger.info("to device")
    model.eval()
    logger.info("eval")
    pdf_embeddings_batch=[]
    blank_check_batch=[]
    for i in range(-(-len(img_list)//batch_sz)):
        image_batch = img_list[i*batch_sz:i*batch_sz+batch_sz]
        logger.info("image_batch "+str(len(image_batch)))
        encoded_inputs = trim_encoded_inputs(processor(image_batch, 
                                                       return_tensors="pt", 
                                                       padding="max_length", 
                                                       truncation=True), limit=encoding_limit)
        logger.info("encoded inputs")
        for k,v in encoded_inputs.items():
            encoded_inputs[k] = v.to(device)
        logger.info("encoded kv")
        with torch.no_grad():
            outputs = model(**encoded_inputs)
        logger.info("outputs")
        pdf_embeddings_batch.append(outputs.hidden_states[-1][:,0].cpu().detach())
        blank_check_batch.append(encoded_inputs['input_ids'][:,1].cpu().detach())
        logger.info("appended")

    pdf_embeddings = torch.cat(pdf_embeddings_batch, 0).numpy()
    logger.info("pdf embeddings")
    blank_check = torch.cat(blank_check_batch, 0).numpy()
    logger.info("blank check")
    logging.info(f"using batch size of {batch_sz} for LayoutLM model")

    return pdf_embeddings, blank_check


def process_png(png_file_list, batch_size, device, encoding_limit):
    """Wrapper for processing pngs

    Args:
        pdf_path (list): list of png file paths
        device (torch.device): the device on which a torch.Tensor is or will be allocated
        encoding_limit (int): Maximum sequence length

    Returns: 
        blank_png_check_list (list): list of 1/0 where 1 indicates corrupted image 
        pdf_embeddings (list): list of numpy.ndarrays of embeddings of images 
    """

    png_embeddings = []
    png_blank_check = []
    image_data = []

    for img_path in png_file_list:
        img = Image.open(img_path)
        image_data.append(img.convert("RGB") ) 

    # LayoutLM inference
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sub_png_embeddings, sub_png_blank_check = batch_inference(image_data, batch_size, device, encoding_limit)
    png_embeddings.append(sub_png_embeddings)
    png_blank_check.extend(list(sub_png_blank_check))
    
    # A list tracking blank images, check batch inference function for further information
    blank_png_check_list = [int(bc==102) for bc in png_blank_check]

    return png_embeddings, blank_png_check_list


def decide_optimal_k(vec_lowd, max_tries):
    """Test different number of clusters and use segregation to determine optimal k
    Args: 
        vec_lowd (np.ndarray): documents embeddings, size=(number of samples, embedding length)
        max_tries (int): number of tries for elbow method
    Returns: 
        optimal_k (int): optimal number of cluster
    """

    # segregation means
    mean_seg_list = []
    for current_k in range(2, min(len(vec_lowd), max_tries)):
        
        kmeans_model = KMeans(n_clusters = current_k, random_state = 0)
        kmeans_model.fit(vec_lowd)
        cluster_label_list = kmeans_model.labels_
        closest_to_centroid, _ = pairwise_distances_argmin_min(kmeans_model.cluster_centers_, vec_lowd)
        
        seg_list = get_segregation(current_k, vec_lowd, closest_to_centroid, cluster_label_list)
        if len(seg_list)==0:
            try:
                # return the next K following the K that brings max segregation
                # K starts from 2, so 2 + 1 = 3
                mean_seg = np.array(mean_seg_list)
                optimal_k = np.argmax(mean_seg)+3
            except ValueError:
                # in cases when sample size is small, single sample cluster could exist,
                # if it does, stop loop and directly return current k as optimal k
                optimal_k = current_k
            return optimal_k
            
        mean_seg_list.append(abs(np.mean(np.array(seg_list))))
    
    # return the next K following the K that brings max segregation
    mean_seg = np.array(mean_seg_list)
    optimal_k = np.argmax(mean_seg)+3

    return optimal_k

def get_segregation(current_k, vec_lowd, closest_to_centroid, cluster_label_list):
    """Given a K, calculate the segregation for each cluster
    Args: 
        current_k (int): number for clusters to test
        vec_lowd (np.ndarray): documents embeddings
        closest_to_centroid (list): a list of centroid ids
        cluster_label_list (list): cluster labels for all embeddings
    Returns: 
        seg_list (list): segregation ratio for each cluster
    """

    seg_list = []
    data_size = len(vec_lowd)
    
    # Check if there exists cluster with less then 2 samples
    cluster_dict = Counter(cluster_label_list)
    cluster_least_freq_id = min(cluster_dict, key=lambda x: cluster_dict[x])
    if cluster_dict[cluster_least_freq_id]<3:
        return seg_list

    for cen_id in range(current_k):
        incluster_id = [i for i in range(data_size) if cluster_label_list[i]==cen_id]
        outcluster_id = [i for i in range(data_size) if cluster_label_list[i]!=cen_id]
        is_in_cluster = vec_lowd[incluster_id]
        is_out_cluster = vec_lowd[outcluster_id]
        is_centroid = vec_lowd[closest_to_centroid[cen_id]].reshape(1,-1) 
            
        w = np.mean(cdist(is_centroid.astype(float), is_in_cluster.astype(float), 'euclidean'))
        b = np.mean(cdist(is_in_cluster.astype(float), is_out_cluster.astype(float), 'euclidean'))
        seg_list.append(abs(w-b)/w)
    
    return seg_list


def cluster_kmeans_elbow(df_embeddings, user_define_k, max_tries, show_tsne=False, use_pca=True,plot_location="/tmp/plot.png"):
    """Kmeans clustering
    Args: 
        df_embeddings (dataframe): embedding dataframe
        user_define_k (int): user defined number of clusters, if not 0, will override elbow method
        max_tries (int): number of tries for elbow method
        show_tsne (boolean): save tsne plot if True
        use_pca (boolean): apply pca on embeddings if True
        plot_location (string): where to save the plot.png to and name overwrite
    Returns: 
        cluster_label_list (list): list of document labels
        confidence_score_list (list): list of document confidence scores
        closest_to_centroid (list): list of ids for centroid documents in the embedding dataframe
    """
    if use_pca== True:
        pca = PCA(n_components=min(len(df_embeddings),128))
        vec_lowd = pca.fit_transform(np.stack(df_embeddings["token0v2"].values))
    else:
        vec_lowd = np.stack(df_embeddings["token0v2"].values)

    # Kmeans requires at least 2 data samples 
    if len(df_embeddings)<3:
        cluster_label_list = [0]*len(df_embeddings)
        confidence_score_list = [0.0]*len(df_embeddings)
        closest_to_centroid = [0]
        return cluster_label_list, confidence_score_list, closest_to_centroid

    # When user_define_k not 0, elbow method will be overridden by the value assigned to user_define_k in the config file
    if user_define_k != 0:
        if user_define_k>len(vec_lowd):
            logging.info("user_define_k larger than number of documents")
            cluster_label_list = [i for i in range(len(df_embeddings))]
            confidence_score_list = [0.0]*len(df_embeddings)
            closest_to_centroid = [i for i in range(len(df_embeddings))]
            return cluster_label_list, confidence_score_list, closest_to_centroid
        elif user_define_k<2:
            logging.info("user_define_k less than 2")
            cluster_label_list = [0]*len(df_embeddings)
            confidence_score_list = [0.0]*len(df_embeddings)
            closest_to_centroid = [0]
            return cluster_label_list, confidence_score_list, closest_to_centroid

        else:
            optimal_k = user_define_k
    else:
        optimal_k = decide_optimal_k(vec_lowd, max_tries)

    kmeans_model = KMeans(n_clusters = optimal_k, random_state = 0).fit(vec_lowd)
    cluster_label_list = kmeans_model.labels_
    closest_to_centroid, _ = pairwise_distances_argmin_min(kmeans_model.cluster_centers_, vec_lowd)
    # A silhouette coefficient is between -1 and 1
    conf_scores = silhouette_samples(vec_lowd, cluster_label_list, metric='euclidean')
    conf_scores = (conf_scores+1)/2
    confidence_score_list = (conf_scores*100).astype(int)/100
    if show_tsne:
        vec_tsne = TSNE(n_components = 2,random_state = 0).fit_transform(vec_lowd)
        plt.figure(figsize=(8, 6))
        plt.scatter(vec_tsne[:,0], vec_tsne[:,1], c=kmeans_model.labels_,cmap=plt.cm.get_cmap('plasma', optimal_k))
        cbar = plt.colorbar(ticks=range(optimal_k), label="Cluster Label")
        cbar.set_ticklabels(range(1, optimal_k + 1))
        plt.clim(-0.5, optimal_k - 0.5)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.axis("off")
        plt.savefig(plot_location)

    return cluster_label_list.tolist(), confidence_score_list.tolist(), closest_to_centroid
    

def pdf2keywords_keybert(pdf_path, processor, keyphrase_ngram_range=(1,1), top_n=10):
    """Extracts keywords from a pdf

    Args:
        pdf_path (str): path to pdf file
        processor (transformers.ProcessorMixin class): text processor object including a LayoutLMv2 feature extractor and a LayoutLMv2 tokenizer
        keyphrase_ngram_range (tuple): a tuple containing length of ngrams
        top_n (int): number of keywords to extract from decoded document
    Returns: 
        keywords (list): list of extracted keywords
    """
    image, corrupt = single_pdf_to_image(pdf_path)
    encoded_inputs = processor(image, padding="max_length", truncation=True)
    decoded_inputs = processor.tokenizer.decode(encoded_inputs.input_ids[0])

    kw_model = KeyBERT(model='all-mpnet-base-v2')
    keywords = kw_model.extract_keywords(decoded_inputs,
                                        keyphrase_ngram_range=keyphrase_ngram_range, 
                                        stop_words='english', 
                                        highlight=False,
                                        use_mmr=True,
                                        top_n=top_n)
    keywords_list= list(dict(keywords).keys())
    # remove special tokens (artifacts from KeyBERT model)
    clean_keywords_list = [keyword for keyword in keywords_list if keyword not in ("cls", "sep")]
    # if no important words, output first word in document.
    if len(keywords_list)==0:
        return decoded_inputs.split(" ")[1]
    return clean_keywords_list


def assign_discriptions(x,centroid_discriptions):
    """Assign cluster discriptions to documents

    Args:
        x (str): cluster label 
        centroid_discriptions (list): list of discriptions of centrod documents
    Returns: 
        centroid discriptions (list/str): list of keywords/discription of a document
    """
    if x=="corrupt":
        return "corrupt"
    elif x=="no_content":
        return "no_content"
    else:
        return list(centroid_discriptions)[int(x)]
