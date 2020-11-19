# dynamic_vehicle_network_clustering
Under different network representation learning (NRL) methods and clustering methods, we cluster the vehicles in the dynamic vehicle network, and compare the clustering performance.

**Paper: Vehicle Trajectory Clustering Based on Dynamic Representation Learning of Internet of Vehicles (https://ieeexplore.ieee.org/document/9115819)**

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

## The commands for training the embedding vectors of vehicles using different NRL methods are listed in cmd_command.txt
