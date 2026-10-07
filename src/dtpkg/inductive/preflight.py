"""Shared inner-validation assignment used by eligibility and model fitting."""
import numpy as np
from sklearn.model_selection import train_test_split

def planned_training_roster(development_pairs, training_seed, val_split=.15):
    """Exactly mirror FusionTrainer._split_train_val with repetition=0.

    The experiment runner calls run_inductive_eval(num_experiments=1) separately
    for each training seed. Positive resampling leaves these negatives fixed.
    """
    train, validation = train_test_split(
        np.arange(len(development_pairs)), test_size=val_split,
        random_state=int(training_seed), stratify=development_pairs.label.to_numpy())
    negatives = development_pairs.iloc[train].query("label == 0")
    return set(negatives.drug1) | set(negatives.drug2), train, validation
