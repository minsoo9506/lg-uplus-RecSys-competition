from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


def DeepFMMakeBaselineData(
    history: pd.DataFrame, meta: pd.DataFrame, profile: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """make baseline data

    Parameters
    ----------
    history : pd.DataFrame
        _description_
    meta : pd.DataFrame
        _description_
    profile : pd.DataFrame
        _description_

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        train_test_data, meta_use, profile_use
    """
    # 중복제거, 사용컬럼 추출
    history_use = (
        history[["profile_id", "log_time", "album_id"]]
        .drop_duplicates(subset=["profile_id", "album_id", "log_time"])
        .sort_values(by=["profile_id", "log_time"])
        .reset_index(drop=True)
        .drop("log_time", axis=1)
    )
    # 중복제거, 사용컬럼 추출
    meta_use = meta[["album_id", "genre_large", "genre_mid"]].drop_duplicates(
        ["album_id"]
    )
    # 중복제거, 사용컬럼 추출
    profile_use = profile[
        [
            "profile_id",
            "sex",
            "age",
            "pr_interest_keyword_cd_1",
            "ch_interest_keyword_cd_1",
        ]
    ]
    # meta, profile 데이터 추가
    history_meta = pd.merge(history_use, meta_use, on="album_id", how="left")
    history_meta_profile = pd.merge(
        history_meta, profile_use, on="profile_id", how="left"
    )
    # 시청 여부 추가
    history_meta_profile["ratings"] = 1

    return history_meta_profile, meta_use, profile_use


class DeepFMTrainTestSplit:
    def __init__(
        self, data: pd.DataFrame, test_size: float = 0.2, random_seed: int = 0
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """split train, test

        Parameters
        ----------
        data : pd.DataFrame
            original data to make train, test dataset
        test_size : float, optional
            _description_, by default 0.2
        random_seed : int, optional
            _description_, by default 0

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            train, test
        """
        self.data = data
        self.test_size = test_size
        self.random_seed = random_seed
        # history
        self.user_to_index = {
            original: idx for idx, original in enumerate(data["profile_id"].unique())
        }
        self.item_to_index = {
            original: idx for idx, original in enumerate(data["album_id"].unique())
        }
        # meta
        self.genre_large_to_index = {
            original: idx for idx, original in enumerate(data["genre_large"].unique())
        }
        self.genre_mid_to_index = {
            original: idx for idx, original in enumerate(data["genre_mid"].unique())
        }
        # profile
        self.sex_to_index = {
            original: idx for idx, original in enumerate(data["sex"].unique())
        }
        self.age_to_index = {
            original: idx for idx, original in enumerate(data["age"].unique())
        }
        self.pr_interest_keyword_cd_1_to_index = {
            original: idx
            for idx, original in enumerate(data["pr_interest_keyword_cd_1"].unique())
        }
        self.ch_interest_keyword_cd_1_to_index = {
            original: idx
            for idx, original in enumerate(data["ch_interest_keyword_cd_1"].unique())
        }

    def split(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """split train, test set

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            train, test
        """
        train, test = train_test_split(
            self.data,
            test_size=self.test_size,
            random_state=self.random_seed,
        )

        return train, test


class DeepFMDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        is_train: bool,
        DeepFMTrainTestSplit: DeepFMTrainTestSplit,
        unique_item: List[int] = None,
        neg_ratio: int = None,
        meta_use: pd.DataFrame = None,
        profile_use: pd.DataFrame = None,
    ):
        """Dataset for DeepFM

        Parameters
        ----------
        data : pd.DataFrame
            _description_
        is_train : bool
            _description_
        DeepFMTrainTestSplit : DeepFMTrainTestSplit
            _description_
        unique_item : List[int], optional
            _description_, by default None
        neg_ratio : int, optional
            _description_, by default None
        meta_use : pd.DataFrame, optional
            _description_, by default None
        profile_use : pd.DataFrame, optional
            _description_, by default None
        """
        super().__init__()

        if is_train:
            neg_samples = np.zeros((data.shape[0] * neg_ratio, data.shape[1]))
            neg_samples_df = pd.DataFrame(neg_samples)
            neg_samples_df.columns = data.columns
            unique_item_set = set(unique_item)

            neg_samples_df = self._negative_sampling(
                data, neg_ratio, meta_use, profile_use, unique_item_set, neg_samples_df
            )

            final_data = pd.concat([data, neg_samples_df])

        else:
            final_data = data

        final_data["profile_id"] = final_data["profile_id"].apply(
            lambda x: DeepFMTrainTestSplit.user_to_index[x]
        )
        final_data["album_id"] = final_data["album_id"].apply(
            lambda x: DeepFMTrainTestSplit.item_to_index[x]
        )

        final_data["genre_large"] = final_data["genre_large"].apply(
            lambda x: DeepFMTrainTestSplit.genre_large_to_index[x]
        )
        final_data["genre_mid"] = final_data["genre_mid"].apply(
            lambda x: DeepFMTrainTestSplit.genre_mid_to_index[x]
        )

        final_data["sex"] = final_data["sex"].apply(
            lambda x: DeepFMTrainTestSplit.sex_to_index[x]
        )
        final_data["age"] = final_data["age"].apply(
            lambda x: DeepFMTrainTestSplit.age_to_index[x]
        )
        final_data["pr_interest_keyword_cd_1"] = final_data[
            "pr_interest_keyword_cd_1"
        ].apply(lambda x: DeepFMTrainTestSplit.pr_interest_keyword_cd_1_to_index[x])
        final_data["ch_interest_keyword_cd_1"] = final_data[
            "ch_interest_keyword_cd_1"
        ].apply(lambda x: DeepFMTrainTestSplit.ch_interest_keyword_cd_1_to_index[x])

        final_data = final_data.to_numpy()
        final_data = final_data.astype(np.intc)
        self.inputs = self.final_data[:, :-1]
        self.targets = self.final_data[:, -1]
        self.field_dims = np.max(self.inputs, axis=0) + 1

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

    def _negative_sampling(
        self,
        train_data: pd.DataFrame,
        neg_ratio: float,
        meta_use: pd.DataFrame,
        profile_use: pd.DataFrame,
        unique_item_set: dict,
        neg_samples_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """negative sampling function

        Parameters
        ----------
        train_data : pd.DataFrame
            _description_
        neg_ratio : float
            _description_
        meta_use : pd.DataFrame
            _description_
        profile_use : pd.DataFrame
            _description_
        unique_item_set : dict
            total unique item set
        neg_samples_df : pd.DataFrame
            _description_

        Returns
        -------
        pd.DataFrame
            negative samplesd result df
        """
        idx = 0
        for id, other_features in train_data.groupby("profile_id"):
            pos_samples = other_features["album_id"].values
            n_neg_samples = len(pos_samples) * neg_ratio
            neg_sample_candidates = list(unique_item_set - set(pos_samples))
            neg_item_ids = np.random.choice(
                neg_sample_candidates,
                min(n_neg_samples, len(neg_sample_candidates)),
                replace=False,
            )
            # 결과 넣기
            neg_samples_df.iloc[idx : idx + n_neg_samples, 0] = id
            neg_samples_df.iloc[idx : idx + n_neg_samples, 1:4] = meta_use.loc[
                meta_use["album_id"].isin(neg_item_ids), :
            ].values
            neg_samples_df.iloc[idx : idx + n_neg_samples, 4:-1] = profile_use.loc[
                profile_use["profile_id"] == id, "sex":
            ].values
            neg_samples_df.iloc[idx : idx + n_neg_samples, -1] = 0
            # idx 수정
            idx += n_neg_samples

        return neg_samples_df
