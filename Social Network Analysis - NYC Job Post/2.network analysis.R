library(igraph)
library(tidyverse)
g = read.graph(file = "C:/Users/a/Desktop/BIA-658/Project/NYC-225-Giant.gml", format = "gml")

# Network-wise Analysis

# degree centralization
centralization.betweenness(g)
#density
edge_density(g,loops = FALSE)
# clustering coefficient
transitivity(g, type = "global")

#homophily
V(g)$PostingType = as.character(job$`Posting Type`[match(V(g)$name,job$Name)])
V(g)$Status = as.character(job$`Full-Time/Part-Time indicator`[match(V(g)$name,job$Name)])
assortativity_nominal(graph = g, types = as.factor(V(g)$PostingType), directed = F)
assortativity_nominal(graph = g, types = as.factor(V(g)$Status), directed = F)
assortativity_degree(graph = g, directed = F)

#shortest path and distance
dis <- distances(g, v = V(g), to = V(g), weights = E(g)$weight, algorithm = "dijkstra")

#diameter
diameter(g, directed = FALSE)
d <- get_diameter(g, directed = FALSE)
diameter <- induced.subgraph(g,vids = d)
write.graph(diameter, file ="diameter.gml", format = "gml")


# Node-wise Analysis

#degree centrality 
degree(g)
degree(g, normalized = TRUE)
# use table to get a degree distribution
table(degree(g))
barplot(table(degree(g)))
# use normalized degree as vertex size
V(g)$size=degree(g, normalized = TRUE)*100
plot(g, vertex.label.cex=1, edge.width = E(g)$weight, layout = layout.fruchterman.reingold(g), rescale=TRUE, asp=0)

#betweenness centrality
betweenness(graph = g, normalized = FALSE)
betweenness(graph = g, normalized = TRUE)
# use normalized betweenness centrality as vertex size
V(g)$size=betweenness(graph = g, normalized = TRUE)*100
plot(g, vertex.label.cex=1, edge.width = E(g)$weight, layout = layout.fruchterman.reingold(g), rescale=TRUE, asp=0)

#closeness centrality
closeness(g)
# use normalized closeness centrality as vertex size
V(g)$size=closeness(graph = g, normalized = TRUE)*50000
plot(g, vertex.label.cex=1, edge.width = E(g)$weight, layout = layout.fruchterman.reingold(g), rescale=TRUE, asp=0)

# eigenvalue centrality
evcent(g)

# Which job is the most important one in our network?
sort(degree(g, normalized = T), decreasing = T)
sort(closeness(g, normalized = T), decreasing = T)
sort(betweenness(g, normalized = T), decreasing = T)
sort(evcent(g)$vector, decreasing = T)

#cliques
cliques(g)
maximal.cliques(g)
a <- largest_cliques(g)
clique1 <- a[[1]]
clique2 <- a[[2]]
clique3 <- a[[3]]
g1 <- induced.subgraph(graph=g,vids=clique1)
g2 <- induced.subgraph(graph=g,vids=clique2)
g3 <- induced.subgraph(graph=g,vids=clique3)
write.graph(g1, file ="largestclique1.gml", format = "gml")
write.graph(g2, file ="largestclique2.gml", format = "gml")
write.graph(g3, file ="largestclique3.gml", format = "gml")

