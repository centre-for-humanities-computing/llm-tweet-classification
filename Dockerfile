FROM nvidia/cuda:12.2.0-devel-ubuntu22.04
RUN apt update
RUN apt install python3 -y
RUN apt install python3-pip -y
RUN apt install git -y
RUN git clone https://github.com/centre-for-humanities-computing/llm-tweet-classification.git 
RUN cd /llm-tweet-classification && pip install -r "requirements.txt"
CMD python3 llm-tweet-classification/llm_classification.py "/config.cfg" > "/logs.txt" 2>&1
