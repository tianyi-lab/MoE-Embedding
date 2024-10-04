from __future__ import annotations

import datasets
import polars as pl

class MultiSubsetLoader:
    def load_data(self, **kwargs):
        """Load dataset containing multiple subsets from HuggingFace hub"""
        if self.data_loaded:
            return

        if hasattr(self, "fast_loading") and self.fast_loading:
            self.fast_load()
        else:
            self.slow_load()

        self.dataset_transform()
        self.data_loaded = True

    def fast_load(self, **kwargs):
        """Load all subsets at once, then group by language with Polars. Using fast loading has two requirements:
        - Each row in the dataset should have a 'lang' feature giving the corresponding language/language pair
        - The datasets must have a 'HS' config that loads all the subsets of the dataset (see https://huggingface.co/docs/datasets/en/repository_structure#configurations)
        """
        self.dataset = {}
        merged_dataset = datasets.load_dataset(
            **self.metadata.dataset
        )  # load "default" subset
        
        for split in merged_dataset.keys():
            # Convert the Hugging Face dataset to a Pandas DataFrame first, then to Polars
            df_split = pl.DataFrame(merged_dataset[split].to_pandas())
            
            # Group by 'lang' using Polars' groupby, then iterate over the groups
            for lang, group in df_split.partition_by("lang", as_dict=True).items():
                if lang in self.hf_subsets:
                    self.dataset.setdefault(lang, {})
                    group = group.drop("lang")  # Drop the 'lang' column
                    # Convert the Polars DataFrame back to a Pandas DataFrame and then to a Hugging Face Dataset
                    self.dataset[lang][split] = datasets.Dataset.from_pandas(group.to_pandas())

        for lang, subset in self.dataset.items():
            self.dataset[lang] = datasets.DatasetDict(subset)


    def slow_load(self, **kwargs):
        """Load each subsets iteratively"""
        self.dataset = {}
        for lang in self.hf_subsets:
            self.dataset[lang] = datasets.load_dataset(
                name=lang,
                **self.metadata.dataset,
            )
