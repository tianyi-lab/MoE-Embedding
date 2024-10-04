from __future__ import annotations

import logging
from typing import Any

import sklearn
import sklearn.cluster
from sklearn import metrics
import numpy as np

from mteb.encoder_interface import Encoder

from .Evaluator import Evaluator
from .model_encode import model_encode

from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


class ClusteringEvaluator(Evaluator):
    def __init__(
        self,
        sentences,
        labels,
        task_name: str | None = None,
        clustering_batch_size: int = 500,
        limit: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if limit is not None:
            sentences = sentences[:limit]
            labels = labels[:limit]
        self.sentences = sentences
        self.labels = labels
        self.clustering_batch_size = clustering_batch_size
        self.task_name = task_name

    def __call__(self, model: Encoder, *, encode_kwargs: dict[str, Any] = {}):
        if "batch_size" not in encode_kwargs:
            encode_kwargs["batch_size"] = 32

        corpus_embeddings, default, mow_rw = model_encode(
            self.sentences,
            model=model,
            prompt_name=self.task_name,
            **encode_kwargs,
        )
        
        if encode_kwargs['do_pca']:
            from sklearn.impute import SimpleImputer
            if np.isnan(corpus_embeddings).any():
                imputer = SimpleImputer(strategy='mean')
                corpus_embeddings = imputer.fit_transform(corpus_embeddings)
                
            if np.isnan(default).any():
                imputer = SimpleImputer(strategy='mean')
                default = imputer.fit_transform(default)

            if np.isnan(mow_rw).any():
                imputer = SimpleImputer(strategy='mean')
                mow_rw = imputer.fit_transform(mow_rw)
                
            pca = PCA(n_components=min(encode_kwargs['pca_dim'], corpus_embeddings.shape[0]))
            corpus_embeddings = pca.fit_transform(corpus_embeddings)
            
            pca = PCA(n_components=min(encode_kwargs['pca_dim'], default.shape[0]))
            default = pca.fit_transform(default)
            
            pca = PCA(n_components=min(encode_kwargs['pca_dim'], mow_rw.shape[0]))
            mow_rw = pca.fit_transform(mow_rw)

        logger.info("Fitting Mini-Batch K-Means model...")
        
        if encode_kwargs['similarity_ensemble']:
            
            clustering_model1 = sklearn.cluster.MiniBatchKMeans(
                n_clusters=len(set(self.labels)),
                batch_size=self.clustering_batch_size,
                n_init="auto",
            )
            clustering_model2 = sklearn.cluster.MiniBatchKMeans(
                n_clusters=len(set(self.labels)),
                batch_size=self.clustering_batch_size,
                n_init="auto",
            )

            clustering_model1.fit(default)
            default_assignment = clustering_model1.labels_
            
            clustering_model2.fit(mow_rw)
            mow_rw_assignment = clustering_model2.labels_
            
            cluster_assignment = (default_assignment + mow_rw_assignment * encode_kwargs['similarity_weights'])/(1+encode_kwargs['similarity_weights'])
        else:
            clustering_model = sklearn.cluster.MiniBatchKMeans(
                n_clusters=len(set(self.labels)),
                batch_size=self.clustering_batch_size,
                n_init="auto",
            )
            clustering_model.fit(corpus_embeddings)
            cluster_assignment = clustering_model.labels_
        

        logger.info("Evaluating...")
        v_measure = metrics.cluster.v_measure_score(self.labels, cluster_assignment)

        return {"v_measure": v_measure}
