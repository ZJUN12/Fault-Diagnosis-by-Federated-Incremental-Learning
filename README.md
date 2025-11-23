# Fault-Diagnosis-by-Federated-Incremental-Learning

This paper has been submitted to ***IEEE Transactions on Cybernetics*** and is currently awaiting the final decision.

This paper proposes an innovative **federated incremental collaborative fault diagnosis framework** for **dynamic data streams**. We focus on a particularly challenging task: performing collaborative diagnosis across **multiple wind farms** under dynamic streaming conditions. The proposed federated learning framework employs a ***three-level hierarchical architecture***, enabling scalable and privacy-preserving collaboration among the wind farms.

The main innovations and contributions of this work can be outlined as follows:
1. Firstly, a **new fault class detection method** is presented to ensure when and where to introduce new fault classes.
2. Secondly, **a balance between the plasticity and stability** of the fault diagnosis model at each wind farm is proposed to alleviate the fading memory problem.
3. Thirdly, **a global model adaptive compensatory** method is presented to address the fading memory issue of the aggregated model caused by heterogeneity.
4. Finally, the proposed method was validated with data from three real-world wind farms in **Hubei, Jiangsu, and Yunnan Provinces, China**. 


***Acknowledgements:*** 
I would like to thank co-author for their valuable guidance and support.  
This project also benefited from the open-source projects and community contributions that made it possible.

Some key references that inspired this work include:  
1. Liu Z, He X, Huang B, et al. Incremental Learning-Enabled Fault Diagnosis of Dynamic Systems: A Comprehensive Review[J]. IEEE Transactions on Cybernetics, 2025.
2. Wang L, Zhang X, Su H, et al. A comprehensive survey of continual learning: Theory, method and application[J]. IEEE transactions on pattern analysis and machine intelligence, 2024, 46(8): 5362-5383.
3. Rebuffi S A, Kolesnikov A, Sperl G, et al. icarl: Incremental classifier and representation learning[C]//Proceedings of the IEEE conference on Computer Vision and Pattern Recognition. 2017: 2001-2010.
4. Dong J, Wang L, Fang Z, et al. Federated class-incremental learning[C]//Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2022: 10164-10173.
5. Wu Y, Chen Y, Wang L, et al. Large scale incremental learning[C]//Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2019: 374-382.


***Prerequisites:***
1. Python == 3.10
2. PyTorch == 2.2
3. numpy == 1.24.4
4. scikit-learn == 1.3.2
