library(tidyverse)
library(igraph)
library(tm)
library(SnowballC)

job <- read_csv("C:/Users/a/Desktop/BIA-658/Project/Data/NYC_Jobs (3).csv")

# basic nlp
job_corpus <- Corpus(VectorSource(job$`Preferred Skills`))
job_corpus <- tm_map(job_corpus, content_transformer(tolower))
job_corpus <- tm_map(job_corpus, removeNumbers)
job_corpus <- tm_map(job_corpus, removePunctuation)
job_corpus <- tm_map(job_corpus, removeWords, c(stopwords("english")))
job_corpus <- tm_map(job_corpus, stripWhitespace)
job_corpus_stemmed <- tm_map(job_corpus, stemDocument)

# generate document-term matrix
job_dtm <- DocumentTermMatrix(job_corpus_stemmed, control = list(bounds = list(global = c(1, Inf))))

# calculate the cosine similarity between the preferred skills of each two jobs
library("lsa")
cosine_sim <- cosine(t(as.matrix(job_dtm)))

# build the network of jobs based on the cosine similarity of "Preferred Skills"
adj_matrix <- ifelse(cosine_sim>0.4, cosine_sim, 0) 
job$Name <- job$`Business Title`
colnames(adj_matrix) <- job$Name
rownames(adj_matrix) <- job$Name
job_graph <- graph_from_adjacency_matrix(adj_matrix, mode = "undirected", weighted = TRUE)
job_graph <- simplify(job_graph)

# assign other attributes of the vertexs
V(job_graph)$Category = as.character(job$`Job Category`[match(V(job_graph)$name,job$Name)])
V(job_graph)$Company = as.character(job$Agency[match(V(job_graph)$name,job$Name)])

# retain the giant component by deleting vertexs whose dgree is 0
job_graph_main <- delete_vertices(job_graph, which(degree(job_graph)==0))


# plot in R
plot(job_graph, vertex.label.cex=1, vertex.size = 1, edge.width = E(job_graph)$weight, layout = layout.fruchterman.reingold(job_graph), rescale=TRUE, asp=0)
plot(job_graph_main, vertex.label.cex=1, vertex.size = 1, edge.width = E(job_graph_main)$weight, layout = layout.fruchterman.reingold(job_graph), rescale=TRUE, asp=0)

# export as gml files
write.graph(job_graph, file ="NYC-225.gml", format = "gml")
write.graph(job_graph_main, file ="NYC-225-Giant.gml", format = "gml")

