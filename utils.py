from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import f1_score
import numpy as np
from sklearn.pipeline import Pipeline

def get_preprocess_std_num(feat_names):
    """Preprocess only the numerical features."""

    def update_num_feats(x):
        if x in feat_names:
            return feat_names.index(x)
    # standardize these variables
    feat_names_num = ["Age", "fe", "Vessels","TSH","ft3","ft4", "Total cholesterol", "HDL","LDL","Triglycerides","Creatinina"]
    num_feat_index = list(map(update_num_feats, feat_names_num))
    num_feat_index = [x for x in num_feat_index if x is not None]
    preprocess_std_num = ColumnTransformer(
                                transformers = [('stand', StandardScaler(), num_feat_index)], 
                                remainder="passthrough",
                                verbose_feature_names_out=False
                            )
    return preprocess_std_num


def datasetSampler(
        model_name,
        model, 
        overSampler, 
        sampling_strategy,
        X_train, 
        y_train, 
        X_valid, 
        y_valid, 
        useUnderSampler =False):
    scores = []
    if useUnderSampler:
        under = RandomUnderSampler(sampling_strategy=sampling_strategy)
        X_train_sample, y_train_sample = under.fit_resample(X_train, y_train)
    else:
        X_train_sample, y_train_sample = X_train, y_train
    
    X_train_sample, y_train_sample = overSampler.fit_resample(X_train_sample, y_train_sample)
    for _ in range(5):
        model.fit(X_train_sample, y_train_sample)
        score = f1_score(y_valid, model.predict(X_valid), average="macro")
        scores.append(score)
    score = np.mean(scores)
    return (score,X_train_sample,y_train_sample, model)


class DebuggablePipeLine(Pipeline):
    def predict_proba(self, X, **predict_proba_params):
        # Uncomment these comments if you want to see if your ColumnTransformer is being applied correctly
        #Xt = X
        # This code is from the source code of Sklearn's Pipeline
        # If your data is transformed correctly there, it'll be also transformed correctly in the super class method.
        """for _, name, transform in self._iter(with_final=False):
                Xt = transform.transform(Xt)
        print(Xt)"""
        return super().predict_proba(X, **predict_proba_params)
    
    @classmethod
    def cast(cls, to_be_casted_obj):
        casted_obj = cls(to_be_casted_obj.steps)
        casted_obj.__dict__ = to_be_casted_obj.__dict__
        return casted_obj


# ==================== Feature naming & cluster mapping ====================

def safe_feature_index(feature_name, feat_names):
    """
    Universal feature mapping that works for all model configurations (18, 23, 27, 32).
    Automatically detects which features are available and maps accordingly.
    """
    
    # Universal mapping dictionary - covers all possible mappings
    name_fixes = {
        # Demographics
        'Gender': 'Gender (Male = 1)',
        
        # Risk factors (cluster names -> CSV names with newlines)
        'Diabetes': 'Diabetes\nHistory of diabetes',
        'Smoke': 'Smoke\nHistory of smoke', 
        'Hypertension': 'Hypertension\nHistory of hypertension',
        'Dyslipidemia': 'Dyslipidemia\nHystory of dyslipidemia',
        
        # Cardiovascular abbreviations -> full names
        'PCI': 'Previous PCI',
        'Previous MI': 'Previous Myocardial Infarction', 
        'Post IDC': 'Post-ischemic Dilated\nCardiomyopathy',
        'LVEF': 'fe',
        'Acute MI': 'Acute Myocardial Infarction',
        
        # Ischemia (special case with newline)
        'Documented resting \nor exertional ischemia': 'Documented resting \nor exertional ischemia',
        
        # Thyroid (Italian -> original, only present in 27/32 feature models)
        'Hypothyroidism': 'Ipotiroidismo',
        'Hyperthyroidism': 'Ipertiroidismo',
        'SCH': 'Subclinical primary hypothyroidism (SCH)',
        'SCT': 'Subclinical primary hyperthyroidism\n(SCT)',
        
        # Features that are the same in cluster and CSV
        'Age': 'Age',
        'Angina': 'Angina', 
        'Angiography': 'Angiography',
        'Vessels': 'Vessels',
        'Previous CABG': 'Previous CABG',
        'Atrial Fibrillation': 'Atrial Fibrillation',
        'TSH': 'TSH',
        'fT3': 'fT3',
        'fT4': 'fT4', 
        'Euthyroid': 'Euthyroid',
        'Low T3': 'Low T3',
        'Total cholesterol': 'Total cholesterol',
        'HDL': 'HDL',
        'LDL': 'LDL',
        'Triglycerides': 'Triglycerides',
        'Creatinina': 'Creatinina',
        'Survive7Y': 'Survive7Y'
    }
    
    # Step 1: Try mapped name if mapping exists
    if feature_name in name_fixes:
        mapped_name = name_fixes[feature_name]
        if mapped_name in feat_names:
            return feat_names.index(mapped_name)
    
    # Step 2: Try exact match
    if feature_name in feat_names:
        return feat_names.index(feature_name)
    
    # Step 3: Try fuzzy matching for edge cases
    feature_lower = feature_name.lower().strip()
    for i, feat in enumerate(feat_names):
        feat_clean = feat.lower().strip().replace('\n', ' ')
        if feature_lower == feat_clean or feature_lower in feat_clean:
            return i
    
    # Step 4: If still not found, show helpful debug info
    print(f"❌ Feature '{feature_name}' not found!")
    if feature_name in name_fixes:
        print(f"   Tried mapping to: '{name_fixes[feature_name]}'")
    
    print(f"   Available features ({len(feat_names)}):")
    for i, feat in enumerate(feat_names):
        print(f"     {i:2}: '{feat}'")
    
    # Try to suggest the closest match
    suggestions = []
    for feat in feat_names:
        if any(word in feat.lower() for word in feature_name.lower().split()):
            suggestions.append(feat)
    
    if suggestions:
        print(f"   💡 Possible matches: {suggestions}")
    
    raise ValueError(f"Feature '{feature_name}' not found in feat_names")


# Optional: Helper function to validate all mappings work
def test_all_mappings(feat_names):
    """Test function to validate mappings work for your specific feat_names"""
    common_cluster_names = [
        'Gender', 'Age', 'Diabetes', 'Smoke', 'Hypertension', 'Dyslipidemia',
        'PCI', 'Previous MI', 'Post IDC', 'LVEF', 'Acute MI', 'Angina',
        'Angiography', 'Vessels', 'Previous CABG', 'Atrial Fibrillation'
    ]
    
    print(f"Testing mappings for {len(feat_names)} feature model...")
    failed = []
    
    for name in common_cluster_names:
        try:
            idx = safe_feature_index(name, feat_names)
            print(f"✅ '{name}' -> '{feat_names[idx]}' (index {idx})")
        except ValueError:
            failed.append(name)
            print(f"❌ '{name}' -> FAILED")
    
    if failed:
        print(f"\n⚠️  Failed mappings: {failed}")
        return False
    else:
        print(f"\n🎉 All mappings successful!")
        return True
# ==================== End Feature naming & cluster mapping ====================