# <img src="https://github.com/OpenGSL/OpenGSL/blob/main/docs/source/img/opengsl.jpg" width="90"> Awesome Graph Structure Learning Papers 
[![Awesome](https://awesome.re/badge-flat2.svg)](https://awesome.re)
>An awesome&amp;curated list of the advanced graph structure learning papers(2023-present) with summaries. We will track the development of GSL and update the list frequently. :pray:**Please star us if you found our repository helpful**.

> \*We have developed [OpenGSL](https://github.com/OpenGSL/OpenGSL), a fair and comprehensive platform to evaluate existing GSL works and facilitate future GSL research. **Please star us and explore our project if you found our repositories helpful**.

## 2024
- **[WWW 2024]** [Self-Guided Robust Graph Structure Refinement](https://dl.acm.org/doi/10.1145/3589334.3645522) propose to extract clean subgraphs and refine the extracted graph structure by leveraging both topological and feature similarity. Additionally, it employ a degree-based group training strategy to address the unbalanced degree distribution.[[Code](https://github.com/yeonjun-in/torch-SG-GSR)]
## 2023
- **[NIPS 2023]** [On the Ability of Graph Neural Networks to Model Interactions Between Vertices](https://proceedings.neurips.cc/paper_files/paper/2023/hash/543ec10715d964122ab7cb15f648772b-Abstract-Conference.html) first propose a novel graph characteristic, the walk index, which reflects interactions between two groups.Then, it develop the Walk Index Sparsification (WIS) algorithm, a graph structure sparsification method that removes noise edges potentially lowering the walk index between the two groups.[[Code](https://github.com/noamrazin/gnn_interactions)]
- **[ICLR 2023]** [Empowering Graph Representation Learning with Test-Time Graph Transformation](https://openreview.net/pdf?id=Lnxl5pr018) incorporate a learnable perturbation matrix with the graph adjacency matrix and adopt a self-supervised contrastive loss as a surrogate to perform optimization at test time. This approach can mitigates distribution shifts between training data and test data.[[Code](https://github.com/ChandlerBang/GTrans)]
- **[NIPS 2023]**[Optimal Block-wise Asymmetric Graph Construction for Graph-based Semi-supervised Learning](https://proceedings.neurips.cc/paper_files/paper/2023/hash/e142fd2b70f10db2543c64bca1417de8-Abstract-Conference.html) construct an asymmetric graph structure that maintains indirect connections from labeled nodes to unlabeled nodes, enhancing the propagation of supervised signals on unlabeled data. Notably, this paper aims to perform Graph Structure Learning (GSL) in the absence of an initial graph structure.
- **[NIPS 2023]**[Latent Graph Inference with Limited Supervision](https://proceedings.neurips.cc/paper_files/paper/2023/hash/67101f97dc23fcc10346091181fff6cb-Abstract-Conference.html) aims to reduce the occurrence of starved nodes(nodes with mutiple unlabeled high-order neighbors) in the graph structure. It construct a similarity matrix between labeled and unlabeled nodes and fuse it with the original graph structure matrix to enhance the propagation of supervised signals to unlabeled data.[[Code](https://github.com/Jianglin954/LGI-LS)]
- **[NIPS 2023]**[Curriculum Learning for Graph Neural Networks: Which Edges Should We Learn First](https://github.com/rollingstonezz/Curriculum_learning_for_GNNs) utilize a Variational Autoencoder (VAE) to recover the graph structure and sort edges using residual loss. Subsequently, it progressively train a GNN by feeding it easier edges(edge with lower residual loss). In this way, the GNN are well-trained and focus on optimizing performance on good structure data.[[Code](https://github.com/rollingstonezz/Curriculum_learning_for_GNNs)]
- **[NIPS 2023]**[Contrastive Graph Structure Learning via Information Bottleneck for Recommendation](https://proceedings.neurips.cc/paper_files/paper/2022/hash/803b9c4a8e4784072fdd791c54d614e2-Abstract-Conference.html) aims to address popularity bias in recommendation systems. First, it introduces a multi-view (node- and edge-drop) graph augmentation technique to eliminate noise structures. Then, it employs an information bottleneck approach to minimize mutual information between original features and augmented features, which can learn invariant information related to labels.[[Code](https://github.com/weicy15/CGI)]
- **[ICML 2023]**[Beyond Homophily: Reconstructing Structure for Graph-agnostic Clustering](https://proceedings.mlr.press/v202/pan23b/pan23b.pdf) aims to adaptively address graph structures with varying levels of homophily. It begins by transforming the graph into homogeneous and heterogeneous graphs. Next, it employs mixed-pass filters to encode graph structure features. Finally, dual encoders are utilized to separately learn attribute and topological features.[[Code](https://github.com/Panern/DGCN)]
- **[KDD 2023]**[PROSE: Graph Structure Learning via Progressive Strategy](https://dl.acm.org/doi/abs/10.1145/3580305.3599476) aims to progressively identify influential nodes using PageRank scores and reconstruct the graph structure by connecting these influential nodes. This approach prioritizes connecting important nodes within graph. [[Code](https://github.com/tigerbunny2023/PROSE)]
- **[KDD 2023]**[GraphGLOW: Universal and Generalizable Structure Learning for Graph Neural Networks](https://dl.acm.org/doi/abs/10.1145/3580305.3599373) This paper aims to learn invariant structural information across diverse graph datasets. The model includes a structure learner and multiple Graph Neural Networks (GNNs) tailored to different graph structures. To streamline structure refinement complexity, the model employs the pivot node trick, avoiding the $O(N^2)$ complexity.[[Code](https://github.com/WtaoZhao/GraphGLOW)]
## Contributors
<!-- readme: collaborators,contributors -start -->
<table>
<tr>
    <td align="center">
        <a href="https://github.com/zzysh12345">
            <img src="https://avatars.githubusercontent.com/u/60538191?v=4" width="120;" alt="hilinxinhui"/>
            <br />
            <sub><b>zzysh12345</b></sub>
        </a>
    </td>
     <td align="center">
        <a href="https://github.com/HZAUerHans">
            <img src="https://avatars.githubusercontent.com/u/71540260?v=4" width="120;" alt="hilinxinhui"/>
            <br />
            <sub><b>HZAUerHans</b></sub>
        </a>
    </td>
</tr>
</table>
<!-- readme: collaborators,contributors -end -->
