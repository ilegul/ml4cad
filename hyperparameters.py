import scipy.stats as stats
hyperparameters = {
    "lr" :{
        'model__penalty': ['l1', 'l2', 'elasticnet'], #type of penalities added, 'elasticnet' means both
        'model__dual': [True, False], # use primal or dual form. Default is True.  
        'model__warm_start': [True, False], # default is False. When set to True, reuse the solution of the previous call to fit as initialization, otherwise, just erase the previous solution. 
        'model__C': stats.randint(1, 10), #default is 1, if the value is larger, then it indicates stronger regularization
        'model__max_iter': stats.randint(50, 500), # Default is 100, Maximum number of iterations taken for the solvers to converge.
        'model__solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'] #default is lbfgs. 
    },
    "svc" :{
        'model__C': stats.randint(100, 600),
        'model__kernel': ['rbf', 'poly', 'sigmoid'],
        'model__degree': stats.randint(5, 200),
        'model__gamma': ['scale', 'auto'], #Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’. scale = 1 / (n_features * X.var()) as value of gamma,
        'model__coef0': stats.uniform(0.0, 1), # It is only significant in ‘poly’ and ‘sigmoid’.
        'model__max_iter': [400, 800, 1200, 1600]
    },
    "knn" :{
        'model__n_neighbors': stats.randint(2, 100),
        'model__weights': ('uniform', 'distance'), # ‘distance’ : weight points by the inverse of their distance. in this case, closer neighbors of a query point will have a greater influence than neighbors which are further away.
        'model__algorithm': ('ball_tree', 'kd_tree'),
        'model__leaf_size': stats.randint(10, 60)
    },
    "rf" :{
        'model__n_estimators': stats.randint(10, 200),
        'model__criterion': ('gini', 'entropy'), # The function to measure the quality of a split. Tree-specific parameter
        'model__min_samples_split': stats.randint(1, 8), #The minimum number of samples required to split an internal node
        'model__min_samples_leaf': stats.randint(1, 5), # The minimum number of samples required to be at a leaf node
        'model__max_features': ('sqrt', 'log2', None), # If “sqrt”, then max_features=sqrt(n_features). If “log2”, then max_features=log2(n_features), If None, then max_features=n_features.
        'model__class_weight': ['balanced', 'balanced_subsample'], 
    },
    "adaboost" :{
        'model__n_estimators': stats.randint(10, 100),
        'model__learning_rate': stats.uniform(0.2, 1)
    },
    "nn" :{
        'model__hidden_layer_sizes': [[stats.randint.rvs(100, 300), stats.randint.rvs(50, 150)], [stats.randint.rvs(50, 300)]], 
        'model__solver': ['sgd', 'adam'], #sgd’ refers to stochastic gradient descent. ‘adam’ refers to a stochastic gradient-based optimizer
        'model__learning_rate_init': stats.uniform(0.0005, 0.005), 
        'model__learning_rate': ('constant', 'adaptive'), 
        'model__alpha': stats.uniform(0, 1), #Strength of the L2 regularization term
        'model__early_stopping': [True],
        'model__max_iter': stats.randint(300, 500),
    },
    "gb" :{
        'model__learning_rate': stats.uniform(0.03, 0.2),
        'model__n_estimators': stats.randint(10, 100),
        'model__max_depth': stats.randint(2, 6),
        'model__max_features': ('sqrt', 'log2', None),  # regularization
        'model__subsample': (0.25, 0.5, 0.75, 1),       # regularization
    },
    "xgb" :{
        'model__booster': ['gbtree', 'gblinear', 'dart'],
        'model__eta': stats.uniform(0.05, 0.5),
        'model__gamma': stats.uniform(0, 0.2),
        'model__max_depth': [2, 3, 4, 6],
        'model__n_estimators': stats.randint(10, 100),
        'model__subsample': [0.25, 0.5, 0.75, 1],     # Stochastic regularization
        'model__lambda': stats.uniform(0.5, 1.5),     # L2 regularization
        'model__alpha': stats.uniform(0, 0.5),        # L1 regularization
        'model__scale_pos_weight': [0.2, 0.4, 0.8, 1, 2],
    }
}