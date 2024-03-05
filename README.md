# VN-net

VN-net is aims to be a fully featured recommendation-focused platform to help people find Visual Novels (VNs) they match their interests. 

Specifically, VN-net is an implementation and application of the xDeepFM [[1]](#1) paper on [VNDB](https://vndb.org/)'s data dumps.

To clarify, VN-net is no substitute for VNDB, but rather a supplementary application that recommends VNs.

Currently, the project is still in the data engineering and modeling phase. The vision is to include a clean and minimal web-interface that lets users input their user-name to generate recommendations. There are 5 different types of recommendations that are planned:
1. What's popular (general)
    - Trending, top overall
2. Just for you
    - xDeepFM top-K
3. Based on this item, ....
    - Item-item similarity
4. Surprise Me!
    - Random sample from xDeepFM
5. Continue where you left off
    - Graph based recommendation. E.g. if steins;gate -> steins;gate 0

## Network Architecture

xDeepFM is a relatively complex neural architecture for CTR that takes inspiration from cross-networks and factorization machines, to model high-order bounded-degree feature interactions on a vector *and* bit level. 

## Citations

<a id="1">[1]</a> Lian, J., Zhou, X., Zhang, F., Chen, Z., Xie, X., & Sun, G. (2018). xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems. In *Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. ACM.*
