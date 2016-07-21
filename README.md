NNHetSeq
=====
NNHetSeq is a Bi-LSTM CRF based package for heterogeneous sequence labeling tasks. The system can be used for POS tagging, Named Entity Recognition, and other sequence labeling tasks. 

Performance
=====

Prerequisition
=====
[LibN3L](https://github.com/SUTDNLP/LibN3L)

Compile
======
* Download [LibN3L](https://github.com/SUTDNLP/LibN3L) library and compile it. 
* Open [CMakeLists.txt](CMakeLists.txt) and change "../LibN3L/" into the directory of your [LibN3L](https://github.com/SUTDNLP/LibN3L) package.  

`cmake .`  
`make`  

