import logging
from aequitas.group import Group
from aequitas.bias import Bias

def run_fairness_audit(df_with_predictions):
    """
    Evaluates model fairness across demographics using Aequitas.
    
    KEY FINDING: The model tends to discriminate against younger, generally 
    healthier individuals where precision and sensitivity are lower.
    """
    g = Group()
    xtab, _ = g.get_crosstabs(df_with_predictions)
    
    b = Bias()
    bias_df = b.get_disparity_predefined_groups(xtab, df_with_predictions, 
                                               ref_groups_dict={'Age':'10-13'})
    
    logging.info("Fairness Audit Complete. Check disparity metrics for Age/Income.")
    return bias_df