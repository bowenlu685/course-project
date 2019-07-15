library(igraph)
library(tidyverse)
g = read.graph(file = "C:/Users/a/Desktop/BIA-658/Project/NYC-225.gml", format = "gml")

#Show the Community Detection for two kinds of algorithms

#1st:edge-betweenness community
community = edge.betweenness.community(g, directed=F)
community$membership
community$modularity
set.seed(1)
plot(g,
     vertex.color = community$membership, vertex.label=NA, vertex.size = log(degree(g) + 1),
     mark.groups = by(seq_along(community$membership), community$membership, invisible),
     layout=layout.fruchterman.reingold)

#2nd,fastgreedy community
community2 = fastgreedy.community(g)
community2$membership
community2$modularity
set.seed(1)
plot(g,
     vertex.color = community2$membership,vertex.label=NA, vertex.size = log(degree(g) + 1),
     mark.groups = by(seq_along(community2$membership), community2$membership, invisible),
     layout=layout.fruchterman.reingold)


#Using the Tree Strucutre to do the Community Detection for two kinds of algorithms

#For the edge-betweenness community
plot(as.dendrogram(community))
#Make the 10 clusters solution Manually
cut_at(community, 10)

#For the fastgreedy community
plot(as.dendrogram(community2))
#Make the 10 clusters solution Manually
cut_at(community2, 10)

