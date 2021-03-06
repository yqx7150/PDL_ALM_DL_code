# PDL_ALM_DL_code  

## Introduction
Predual dictionary learning (PDL) / augmented Lagrangian multi-scale dictionary learning(ALM-DL)  
The Code is created based on the method described in the following papers:   
This idea was originally motivated from the predual dictionary learning (PDL)【1】  
Later, we found that it also can be derived from the view of augmented Lagrangian【2】  
【2】is the advanced version, termed ALM-DL:augmented Lagrangian multi-scale dictionary learning    
【1】Qiegen Liu, Shanshan Wang, Jianhua Luo. [A novel predual dictionary learning algorithm. Journal of Visual Communication and Image Representation](https://ac.els-cdn.com/S1047320311001246/1-s2.0-S1047320311001246-main.pdf?_tid=a5cd5f8a-7164-42cd-8668-86d652d89f3a&acdnat=1535963140_7ad392b98a43e181cd32cdadfcfca757), 2012, 23(1): 182-193.    
【2】Qiegen Liu, Jianhua Luo, Shanshan Wang, Moyan Xiao and Meng Ye. [An augmented Lagrangian Multi-scale Dictionary Learning Algorithm. EURASIP Journal on Advances in Signal Processing 2011](https://link.springer.com/article/10.1186/1687-6180-2011-58), 2011:58 doi:10.1186/1687-6180-2011-58.  

## Some results
The learned dictionaries in the iterative process by PDL.    
![PDL_result](/fig/PDL_result.jpg)   

The learned dictionaries in the iterative process by K-SVD.   
![K-SVD_result.jpg](/fig/K-SVD_result.jpg)  


## Other Related Projects
  * Adaptive dictionary learning in sparse gradient domain for image recovery [<font size=5>**[Paper]**</font>](https://ieeexplore.ieee.org/document/6578193/)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/GradDL) 

  * Highly undersampled magnetic resonance image reconstruction using two-level Bregman method with dictionary updating [<font size=5>**[Paper]**</font>](https://ieeexplore.ieee.org/document/6492252)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/TBMDU) 
  
  * Field-of-Experts Filters Guided Tensor Completion [<font size=5>**[Paper]**</font>](https://ieeexplore.ieee.org/document/8291751/similar#similar)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/FoE_STDC)   [<font size=5>**[Slide]**</font>](https://github.com/yqx7150/EDAEPRec/tree/master/Slide)
  
  * Synthesis-analysis deconvolutional network for compressed sensing [<font size=5>**[Paper]**</font>](https://ieeexplore.ieee.org/document/8296620)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/SADN)
  
  * Convolutional Sparse Coding in Gradient Domain for MRI Reconstruction [<font size=5>**[Paper]**</font>](http://html.rhhz.net/ZDHXBZWB/html/2017-10-1841.htm)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/GradCSC)
  
  * Sparse and dense hybrid representation via subspace modeling for dynamic MRI [<font size=5>**[Paper]**</font>](https://www.sciencedirect.com/science/article/abs/pii/S089561111730006X)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/SDR)
