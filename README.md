# Software Similarity Dataset

This project contains the source code for processing raw software into graph representation as well as calculating the similarity score between two software. Dataset is here.

# Description


## GraphRep
This Section introduce the process of how to convert and clean the raw softrware into a embedded graph representation dataset for software similarity.

### Raw Data Obtain
For obtain the dataset, we use two essential tools: Somef and Inspect4py. The raw data is obtained by the github links provided by [Paper With Code](https://paperswithcode.com/).

- **[Somef](https://github.com/KnowledgeCaptureAndDiscovery/somef)**: Automatically extracting relevant information from
readme files and download software.
- **[Inspect4py](https://github.com/SoftwareUnderstanding/inspect4py)**: Inspect python software and extract all info(call
information).

### Pre_Process
The folder of Pre_Process is aim to process the raw dataset into graph representation.
- **get_files_from_software.py**: Get all the functions within one software with unique index name. 
  - Input: Result from Somef and Inspect4py
  - Output: dir_info.json
  - Format: [[python file name: python file location]]
- **process_paperwithcode.py**: Get all the descritive data provided by paperwithcode to the according software
  - Input: paperwithcode dataset
  - Output:  repo2edu.json
  - Format: {"reference": Str， "abstract": Str, "task": List of Str, "author": Str, "url":link, "repo": Str}
- **data_process_to_graph.py**: Process the data with dir_info.json and repo2edu.json into graph representation.
  - Input: dir_info.json & repo2edu.json & Inpsect4py Result
  - Output: Graph Representation
  - Format:
    - line: List <this function line index within the file [num_1, num_2]>
    - call_info: List <all called functions and methods in this function(in and out this repo)>
    - type: Str <function or methods(from a class)>
    - repo_call_info: List <all called function within this repo, if none called --> ["None"]>
    - code_tokens: List <tokenized code of this function>


### SimScore
This section is for getting the numerical similarity metric between two software using pre-train models provided by [Huggingface](https://huggingface.co/). We have used SentenceBert, MiniLM and TSDAE.

- **get_final_abstract.py**: Get a json file with all the abstract according to all the software
  - Input: All software & repo2edu.json
  - Output: matching_data.json
- **select_model_embedding.py**: Use different models for embed the abstract for each software
- **CalSimScore.py**: Calculate the cosine similarity between embedded abstract

### ae_process
This section is dedicated for using autoencoder model for dealing with different length of functions(nodes) in the graph representation of software, an example is givne [Github_Issue](https://github.com/pyg-team/pytorch_geometric/issues/1950)

- **pre_ae_data.py**: this file use pre-trained model for encode the code_tokens into numerical embedding, within this file, we applied the [UniXcoder](https://arxiv.org/abs/2203.03850) model.
  - Input: final_data
  - Output: ae_data
- **get_ae_training_data.py**: sample random 20000 UniXcoder embedded functions for training our autoencoder model. The length of each function length is set to 1024.
  - Input: ae_data
  - Output: data_1024
- **flat_ae_encode.py**: This .py file trained the autoencoder model use data from data_1024 for future encode all functions within the dataset.
  - Input: data_1024
  - Output: A Autoencode Model
- **ae_process_data.py**:This File use the trained model from flat_ae_encoder.py, process the entire dataset with autoencoder model. If the software(bert embedded) is over 300MB, we abandoned the software due to significance in size.
  - Input: Autoencoder Model & ae_data
  - Output: post_process(software count: 2001)
