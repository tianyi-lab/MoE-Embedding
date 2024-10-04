from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics.pairwise import (
    paired_cosine_distances,
    paired_euclidean_distances,
    paired_manhattan_distances,
)

from mteb.encoder_interface import Encoder, EncoderWithSimilarity

from .Evaluator import Evaluator
from .model_encode import model_encode

logger = logging.getLogger(__name__)

from sklearn.decomposition import PCA


class STSEvaluator(Evaluator):
    def __init__(
        self,
        sentences1,
        sentences2,
        gold_scores,
        task_name: str | None = None,
        limit: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if limit is not None:
            sentences1 = sentences1[:limit]
            sentences2 = sentences2[:limit]
            gold_scores = gold_scores[:limit]
        self.sentences1 = sentences1
        self.sentences2 = sentences2
        self.gold_scores = gold_scores
        self.task_name = task_name

    def __call__(
        self,
        model: Encoder | EncoderWithSimilarity,
        *,
        encode_kwargs: dict[str, Any] = {},
    ):
        embeddings1, default1, moe_rw1 = model_encode(
            self.sentences1, model=model, prompt_name=self.task_name, **encode_kwargs
        )
        embeddings2, default2, moe_rw2 = model_encode(
            self.sentences2, model=model, prompt_name=self.task_name, **encode_kwargs
        )
        
        if encode_kwargs['do_pca']: 
            combined_enc = np.concatenate([embeddings1, embeddings2], axis=0)
            pca = PCA(n_components=min(encode_kwargs['pca_dim'], combined_enc.shape[0]))
            pca_result = pca.fit_transform(combined_enc)
            embeddings1 = pca_result[:embeddings1.shape[0], :]
            embeddings2 = pca_result[embeddings1.shape[0]:, :]
            
            if encode_kwargs['similarity_ensemble']:
                combined_enc = np.concatenate([default1, default2], axis=0)
                pca = PCA(n_components=min(encode_kwargs['pca_dim'], combined_enc.shape[0]))
                pca_result = pca.fit_transform(combined_enc)
                default1 = pca_result[:default1.shape[0], :]
                default2 = pca_result[default1.shape[0]:, :]
                
                combined_enc = np.concatenate([moe_rw1, moe_rw2], axis=0)
                pca = PCA(n_components=min(encode_kwargs['pca_dim'], combined_enc.shape[0]))
                pca_result = pca.fit_transform(combined_enc)
                moe_rw1 = pca_result[:moe_rw1.shape[0], :]
                moe_rw2 = pca_result[moe_rw1.shape[0]:, :]

        logger.info("Evaluating...")
        cosine_scores = 1 - (paired_cosine_distances(embeddings1, embeddings2))
        cosine_scores_main = 1 - (paired_cosine_distances(default1, default2))
        cosine_scores_aux = 1 - (paired_cosine_distances(moe_rw1, moe_rw2))
         
        manhattan_distances = -paired_manhattan_distances(embeddings1, embeddings2)
        euclidean_distances = -paired_euclidean_distances(embeddings1, embeddings2)

        cosine_pearson, _ = pearsonr(self.gold_scores, cosine_scores)
        
        cosine_spearman_main, _ = spearmanr(self.gold_scores, cosine_scores_main)
        cosine_spearman_aux, _ = spearmanr(self.gold_scores, cosine_scores_aux)
        
        if encode_kwargs['similarity_ensemble']:
            cosine_spearman, _ = spearmanr(self.gold_scores, [cosine_scores_main[i] + cosine_scores_aux[i] * encode_kwargs['similarity_weights'] for i in range(len(cosine_scores_aux))])
        else:
            cosine_spearman, _ = spearmanr(self.gold_scores, cosine_scores)

        manhatten_pearson, _ = pearsonr(self.gold_scores, manhattan_distances)
        manhatten_spearman, _ = spearmanr(self.gold_scores, manhattan_distances)

        euclidean_pearson, _ = pearsonr(self.gold_scores, euclidean_distances)
        euclidean_spearman, _ = spearmanr(self.gold_scores, euclidean_distances)

        similarity_scores = None
        
        if hasattr(model, "similarity_pairwise"):
            similarity_scores = model.similarity_pairwise(embeddings1, embeddings2)  # type: ignore
        elif hasattr(model, "similarity"):
            _similarity_scores = [
                float(model.similarity(e1, e2))  # type: ignore
                for e1, e2 in zip(embeddings1, embeddings2)
            ]
            similarity_scores = np.array(_similarity_scores)

        if similarity_scores is not None:
            pearson = pearsonr(self.gold_scores, similarity_scores)
            spearman = spearmanr(self.gold_scores, similarity_scores)
        else:
            # if model does not have a similarity function, we assume the cosine similarity
            pearson = cosine_pearson
            if encode_kwargs['similarity_ensemble']:
                spearman = cosine_spearman_main + cosine_spearman_aux*encode_kwargs['similarity_weights']
            else:
                spearman = cosine_spearman
            

        return {
            # using the models own similarity score
            "pearson": pearson,
            "spearman": spearman,
            # generic similarity scores
            "cosine_pearson": cosine_pearson,
            "cosine_spearman": cosine_spearman,
            "manhattan_pearson": manhatten_pearson,
            "manhattan_spearman": manhatten_spearman,
            "euclidean_pearson": euclidean_pearson,
            "euclidean_spearman": euclidean_spearman,
        }
