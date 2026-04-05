import pandas as pd

class FeatureStore:
    """
    A centralized class for managing complex feature calculation and retrieval. 
    
    Key Design Concepts:
    - On-Demand Evaluation: Features calculated only when needed and were cached to prevent redundant computation.
    - Automated Dependency Resolution: Automatically manage complicated dependencies between features by the concept of Directed Acyclic Graph (DAG).
    - Modular & Extensible: New features can be easily added by defining a `_compute_<feature_name>` method.
    
    Definition for Levels:
    - Level 0 features depends on no one (their values are given).
    - Level 1 features depends on level 0 features only (no dependence on other level 1 features).
    - Level 2 features depends on level 0 or 1 features only (no dependence on other level 2 features).    
    
    Minor Design Choices:
    - DataFrame Output: The return of get_ftr is intentionally set to be a dataframe for consistency.
    - Series Internal Output: The internal output is set to series for easier/efficient internal computation. 
    """
    given_features = {'cycle', 
                      'nh3_pulse_time', 
                      'nh3_purge_time', 
                      'ticl4_pulse_time', 
                      'ticl4_purge_time',
                      }
    
    def __init__(self, df: pd.DataFrame, model_descriptor: dict):
        self.df = df.copy() # level 0 features given at initialization
        self.model_descriptor = model_descriptor # some features require a model
    
    # a list of currently availabe (calculated or stored) features
    @property
    def available(self) -> list[str]:
        return list(self.df.columns)
    
    def _get_ftr(self, ftr_name: str) -> pd.Series:
        # return as pandas series for easier computation (like df['A']+df['B'])
        
        # Cache found
        if ftr_name in self.df.columns:
            return self.df[ftr_name]

        # Cache missed - polynomial features
        if ' ' in ftr_name or '^' in ftr_name: 
            self.df[ftr_name] = self._get_polynomial_ftr(ftr_name)

        # Cache missed - non-polynomial features (' ' and '^' not allowed in the feature name )
        else: 
            if ftr_name in self.given_features:
                raise ValueError(f'Missing given value for the feature: {ftr_name}')
            ftr_compute_method = getattr(self, f'_compute_{ftr_name}', None)
            if ftr_compute_method is None:            
                raise NotImplementedError(f'Missing compute method for the feature: {ftr_name}')
            self.df[ftr_name] = ftr_compute_method()
        return self.df[ftr_name]
    
    def _get_polynomial_ftr(self, poly_ftr_name: str) -> pd.Series:
        decomposed_ftrs_with_order = list(poly_ftr_name.split(' '))
        result = 1
        for ftr_with_order in decomposed_ftrs_with_order:
            if '^' in ftr_with_order:
                ftr_name, order = ftr_with_order.split('^')
            else:
                ftr_name, order = ftr_with_order, 1
            result *= self._get_ftr(ftr_name)**float(order)
        return result
    
    def get_ftr(self, ftr_name: str) -> pd.DataFrame:
        self._get_ftr(ftr_name)
        # forced DataFrame return by double []
        return self.df[[ftr_name]]
    
    # multiple feature version for get_ftr
    def get_ftrs(self, ftr_names: list[str]) -> pd.DataFrame:
        for ftr_name in ftr_names:
            self._get_ftr(ftr_name)
        return self.df[ftr_names]

    # level 1 features (alphabetical order)
    def _compute_total_process_time(self) -> pd.Series:
        result = (
                  (self._get_ftr('ticl4_pulse_time') + 
                   self._get_ftr('ticl4_purge_time') + 
                   self._get_ftr('nh3_pulse_time') + 
                   self._get_ftr('nh3_purge_time')
                  )*self._get_ftr('cycle')+90
                 )/60
        return result


    # level 2 features
    def _compute_intermixing(self):
        pass

    


if __name__ == "__main__":
    # Example usage:
    sample_df = pd.DataFrame({'ticl4_pulse_time': [10, 3, 4,], 
                              'ticl4_purge_time': [15, 10, 3], 
                              'nh3_pulse_time': [5, 2, 2], 
                              'nh3_purge_time': [10,7, 8], 
                              'cycle': [1, 8, 2]})
    ftrs = FeatureStore(sample_df, None)
    print("Features:", ftrs.available)
    print('Total process time:', ftrs.get_ftr('total_process_time'))
    print('cycle^2 Total process time:', ftrs.get_ftr('cycle total_process_time^2'))



