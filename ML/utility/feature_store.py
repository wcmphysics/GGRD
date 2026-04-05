import pandas as pd

class FeatureStore:
    """
    A centralized class for managing complex feature calculation and retrival. 
    
    Key Design Concepts:
    - On-Demand Evaluation: Features calculated only when needed and were cached to prevent redundant computation.
    - Automated Dependency Resolution: Automatically manage complicate dependencies between features by the concept of Directed Acyclic Graph (DAG).
    - Modular & Extensible: New features can be easily added by defining a `_compute_<feature_name>` method.
    
    Definition for Levels:
    - Level 0 features depends on no one (their values are given).
    - Level 1 features depends on level 0 features only (no dependence on other level 1 features).
    - Level 2 features depends on level 0 or 1 features only (no dependence on other level 2 features).    
    
    Minor Design Choices:
    - DataFrame Output: The return of get_ftr is intentionally set to be a dataframe for cosnistency.
    - Seires Internal Output: The internal output is set to series for easier/efficient internal compuation. 
    """
    def __init__(self, df: pd.DataFrame, model_descriptor: dict):
        self.df = df.copy() # level 0 features given at initialization
        self.model_descriptor = model_descriptor # some features require a model
        self.available = list(self.df.columns) # a list of currently avaialbe (calcualted or stored) features
    
    def _get_ftr(self, ftr_name: str) -> pd.Series:
        if ftr_name not in self.df.columns:
            ftr_compute_method = getattr(self, f'_compute_{ftr_name}', None)
            if ftr_compute_method is None:            
                raise Exception(f'Missing compute method for the feature: {ftr_name}')
            self.df[ftr_name] = ftr_compute_method()
        # return as pandas series for easier computation (like df['A']+df['B'])
        return self.df[ftr_name]
    
    def get_ftr(self, ftr_name: str) -> pd.DataFrame:
        self._get_ftr(ftr_name)
        # forced DataFrame return by double []
        return self.df[[ftr_name]]
    
    # multiple feature version for get_ftr
    def get_ftrs(self, ftr_names: list[str]) -> pd.DataFrame:
        for ftr_name in ftr_names:
            self._get_ftr(ftr_name)
        return self.df[ftr_names]

    # level 0 features
    # function for features that should be given a value by default at the initialization
    # no check was run at initialization since some features like cycle is not strickly level 0 feature
    def _compute_default_value(self, ftr_name:str) -> pd.Series:
        if ftr_name not in self.df.columns:
            raise Exception(f'Missing default value for the feature: {ftr_name}')
        return self.df[ftr_name]
    
    def _compute_ticl4_pulse_time(self):
        return self._compute_default_value('ticl4_pulse_time')
    
    def _compute_ticl4_purge_time(self):
        return self._compute_default_value('ticl4_purge_time')

    def _compute_nh3_pulse_time(self):
        return self._compute_default_value('nh3_pulse_time')     

    def _compute_nh3_purge_time(self):
        return self._compute_default_value('nh3_purge_time') 
    
    def _compute_cycle(self):
        return self._compute_default_value('cycle') 
    
    # level 1 features
    def _compute_total_process_time(self):
        if 'total_process_time' not in self.df.columns:
            self.df['total_process_time'] = ((self._compute_ticl4_pulse_time() + 
                                              self._compute_ticl4_purge_time() + 
                                              self._compute_nh3_pulse_time() + 
                                              self._compute_nh3_purge_time()
                                             )*self._compute_cycle()+90
                                            )/60
        return self.df['total_process_time']


    # level 2 features
    def _compute_intermixing(self):
        pass

    


if __name__ == "__main__":
    # Example usage:
    sample_df = pd.DataFrame({'ticl4_pulse_time': [10], 
                              'ticl4_purge_time': [15], 
                              'nh3_pulse_time': [5,], 
                              'nh3_purge_time': [10,], 
                              'cycle': [1,]})
    ftrs = FeatureStore(sample_df, None)
    print("Features:", ftrs.available)
    print('Total process time:', ftrs.get_ftr('total_process_time'))



