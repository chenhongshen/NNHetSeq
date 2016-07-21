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

Usage
=====
cd example/
1. run stacking model 
	a. run base model and preserve the parameters using corpus A as command in `run_base.sh`.
	b. then, run stacking model as command in `run_stack.sh` using corpus B, and initialize the stacking model with the base model parameters.
2. run multi-view coupling model as command in `run_couple.sh`
	directly set two resources of corpora (corpus A and corpus B) and run it
3. run integrated stacking and multi-view coupling model
	a. run base model and preserve the parameters using corpus A as command in `run_base.sh`..
	b. run integrated model using corpus A and corpus B as command in `run_couplestack.sh`, and initialize the integrated model with the base model parameters.
	
Notes:
1. NN stacking model (and the integrated model) convergent faster than multi-view coupling model based on our observations.
