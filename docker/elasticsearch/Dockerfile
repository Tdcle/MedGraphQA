ARG ELASTIC_VERSION=8.17.2
FROM docker.elastic.co/elasticsearch/elasticsearch:${ELASTIC_VERSION}

ARG ELASTIC_VERSION=8.17.2
RUN elasticsearch-plugin install --batch https://get.infini.cloud/elasticsearch/analysis-ik/${ELASTIC_VERSION}
