# dynamic_vehicle_network_clustering
Under different network representation learning (NRL) methods and clustering methods, we cluster the vehicles in the dynamic vehicle network, and compare the clustering performance.

# NRL Methods
* DynWalks
* DeepWalk
* LINE
* node2vec
* GraRep

DynWalks: https://github.com/houchengbin/GloDyNE.

We use the code of the other three NRL methods in OpenNE to learn the embedding vectors of vehicles in the dynamic vehicle network.
OpenNE: https://github.com/thunlp/OpenNE

# Clustering methods
* KMeans
* HierarchicalClustering
* SpectralClustering
* GaussianMixture
* KMedoids

# Metrics
* Silhouette Score
* Dunn Validity Index
* Davies Bouldin Score
* Calinski Harabaz Score
