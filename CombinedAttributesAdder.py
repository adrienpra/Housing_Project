from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
        self.rooms_ix = 3
        self.bedrooms_ix = 4
        self.population_ix = 5
        self.households_ix = 6

    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        rooms_per_households = X[:, self.rooms_ix] / X[:, self.households_ix]
        population_per_households = X[:, self.population_ix] / X[:, self.households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, self.bedrooms_ix] / X[:, self.rooms_ix]
            return np.c_[X, rooms_per_households, population_per_households, bedrooms_per_room] #Translates slice objects to concatenation along the second axis.
        else:
            return np.c_[X, rooms_per_households, population_per_households]
