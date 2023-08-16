FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04


RUN sudo apt update
RUN sudo apt install python3
RUN sudo apt install python3-pip
RUN git clone https://github.com/centre-for-humanities-computing/llm-tweet-classification.git 
RUN cd /llm-tweet-classification && pip install -r "requirements.txt"

CMD python3 llm-tweet-classification/llm_classification.py --config "/in_out/" --in_file "/in_out/labelled_data.csv"