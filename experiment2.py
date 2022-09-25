import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib.ticker import StrMethodFormatter
from imblearn.over_sampling import RandomOverSampler

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_validate
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve

from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
import time as tm

from os import path



def decision_tree(X_train, X_test, y_train, y_test):


    learner_name = "Decision Tree"
    print(learner_name)
    learner = DecisionTreeClassifier(splitter='random',max_depth=10,ccp_alpha=0.001)

    t1 = tm.time()
    learner.fit(X_train, y_train)  # 0. Train

    t2 = tm.time()


    t3 = tm.time()
    y_pred = learner.predict(X_test) # 4. Test (Actual test/ Out sample)
    t4 = tm.time()
    y_pred_in = learner.predict(X_train) # 6. Test (On training set itself/ In sample)

    ###########################
    # Clock Times
    dt_train_time = t2 - t1 # Train
    dt_query_time = t4 - t3 # Test/ Query
    print("Training time:", dt_train_time)
    print("Testing time:", dt_query_time)

    ###########################
    ###########################
    # Accuracy/confusion matrix
    test_accuracy=accuracy_score(y_test, y_pred)
    test_recall=recall_score(y_test, y_pred, pos_label=1) # +ve class label is "2" from {1,2} or "1" from {0,1}
    print("Accuracy",test_accuracy)
    print("Recall",test_recall)
    print('\n')


    cf_matrix = confusion_matrix(y_test, y_pred)
    ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues')
    ax.set_title('Confusion Matrix\n');
    ax.set_xlabel('Predicted Values')
    ax.set_ylabel('Actual Values');
    ax.xaxis.set_ticklabels(['Longer', 'Shorter']) # T/F
    ax.yaxis.set_ticklabels(['Longer', 'Shorter']) # T/F
    plt.xticks(rotation='horizontal')
    plt.savefig('2Chart_dt_confusion.png')     # save plot
    # plt.show()
    plt.close()


    ###################################################################################
    # Vailidation curve/ Hyper parameter tuning #1
    path = DecisionTreeClassifier().cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities

    param="ccp_alpha" # pruning ######
    param_range = ccp_alphas
    param_range = param_range[param_range >= 0.0005]
    param_range = param_range[param_range <= 0.013]
    scoring_metric = 'recall'  # // accuracy, precision
    train_scores_vc1, test_scores_vc1 = \
        validation_curve(estimator=DecisionTreeClassifier(),
                        X=X_train,
                        y=y_train,
                        param_name=param,
                        param_range=param_range,
                        cv=4,
                        scoring=scoring_metric,
                        n_jobs=2,)

    train_scores_mean_vc1 = np.mean(train_scores_vc1, axis=1)
    test_scores_mean_vc1 = np.mean(test_scores_vc1, axis=1)
    df = pd.DataFrame({'Complexity Parameter': param_range, 'Training Score': train_scores_mean_vc1, 'Testing Score': test_scores_mean_vc1})



    df.plot(x=0, y=[1,2], kind='line',logx=True) # Positions of cols
    plt.ylabel(scoring_metric.capitalize()) #("Accuracy")
    plt.xlabel("Complexity Parameter")
    plt.suptitle("Validation Curve"+': '+learner_name)
    plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.3f}'))  # 3 decimal places
    plt.grid(True)
    plt.savefig('2Chart_dt_VC1.png')     # save plot
    # plt.show()
    plt.close()





    ###################################################################################
    ###################################################################################
    # # Vailidation curve/ Hyper parameter tuning #2
    # depth = learner.get_depth()
    ccp_alphas, impurities = path.ccp_alphas, path.impurities

    param="max_depth" # depth
    param_range = range(1,25,1) # [2,4,6,8,10,12,14,16]
    scoring_metric = 'recall'  # // accuracy, precision
    train_scores_vc2, test_scores_vc2 = \
        validation_curve(estimator=DecisionTreeClassifier(),
                        X=X_train,
                        y=y_train,
                        param_name=param,
                        param_range=param_range,
                        cv=4,
                        scoring=scoring_metric,
                        n_jobs=2,)

    train_scores_mean_vc2 = np.mean(train_scores_vc2, axis=1)
    test_scores_mean_vc2 = np.mean(test_scores_vc2, axis=1)
    df = pd.DataFrame({'Maximum Depth Parameter': param_range, 'Training Score': train_scores_mean_vc2, 'Testing Score': test_scores_mean_vc2})


    df.plot(x=0, y=[1,2], kind='line',logx=False) # Positions of cols
    plt.ylabel(scoring_metric.capitalize()) #("Accuracy")
    plt.xlabel("Maximum Depth Parameter")
    plt.suptitle("Validation Curve"+': '+learner_name)

    plt.grid(True)
    plt.savefig('2Chart_dt_VC2.png')     # save plot
    # plt.show()
    plt.close()



    ###################################################################################
    ###################################################################################
    ###################################################################################
    # Learning Curve
    scoring_metric = 'recall'  # // accuracy, precision

    train_sizes_lc, train_scores_lc, valid_scores_lc, fit_times_lc, _ = \
        learning_curve(estimator=DecisionTreeClassifier(splitter='random',max_depth=10,ccp_alpha=0.001),
                        X=X_train,
                        y=y_train,
                        scoring=scoring_metric,
                        cv=4,
                        n_jobs=2,
                        train_sizes=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                        return_times=True,)

    train_scores_mean = np.mean(train_scores_lc, axis=1)
    valid_scores_mean = np.mean(valid_scores_lc, axis=1)
    fit_times_mean = np.mean(fit_times_lc, axis=1)

    df = pd.DataFrame({'Sample Size': train_sizes_lc, 'Training Score': train_scores_mean, 'Cross-validation Score': valid_scores_mean})



    df.plot(x=0, y=[1,2]) # Positions of cols
    plt.ylabel(scoring_metric.capitalize()) #("Accuracy")
    plt.xlabel("Sample Size")
    plt.suptitle("Learning Curve"+': '+learner_name)
    plt.grid(True)
    plt.savefig('2Chart_dt_LC.png')     # save plot
    # plt.show()
    plt.close()






def neural_network(X_train, X_test, y_train, y_test):

    learner_name = "Neural Network"
    print(learner_name)
    learner = MLPClassifier(hidden_layer_sizes=(30,5), learning_rate_init=0.000001, max_iter=300)

    t1 = tm.time()
    learner.fit(X_train, y_train)  # 0. Train
    t2 = tm.time()

    t3 = tm.time()
    y_pred = learner.predict(X_test) # 4. Test (Actual test/ Out sample)
    t4 = tm.time()

    y_pred_in = learner.predict(X_train) # 6. Test (On training set itself/ In sample)


    ###########################
    # Clock Times
    dt_train_time = t2 - t1 # Train
    dt_query_time = t4 - t3 # Test/ Query
    print("Training time:", dt_train_time)
    print("Testing time:", dt_query_time)

    ###########################
    ###########################
    # Accuracy/confusion matrix
    test_accuracy=accuracy_score(y_test, y_pred)
    test_recall=recall_score(y_test, y_pred, pos_label=1) # +ve class label is "2" from {1,2} or "1" from {0,1}
    print("Accuracy",test_accuracy)
    print("Recall",test_recall)
    print('\n')


    ###################################################################################
    # Vailidation curve/ Hyper parameter tuning #1
    vc1_name = 'Hidden Layers'                                 # <<
    param="hidden_layer_sizes"                                      # <<
    param_range = [(8,5),(15,5), (20,5), (30,5), (40,5), (55,5),]                             # <<
                          # <<
    scoring_metric = 'recall'  # // accuracy, precision
    train_scores_vc1, test_scores_vc1 = \
        validation_curve(estimator=MLPClassifier(),                # <<
                        X=X_train,
                        y=y_train,
                        param_name=param,
                        param_range=param_range,
                        cv=4,
                        scoring=scoring_metric,
                        n_jobs=2,)

    train_scores_mean_vc1 = np.mean(train_scores_vc1, axis=1)
    test_scores_mean_vc1 = np.mean(test_scores_vc1, axis=1)
    df = pd.DataFrame({vc1_name: param_range, 'Training Score': train_scores_mean_vc1, 'Testing Score': test_scores_mean_vc1})



    df.plot(x=0, y=[1,2], kind='line',logx=False) # Positions of cols           # <<
    plt.ylabel(scoring_metric.capitalize()) #("Accuracy")
    plt.suptitle("Validation Curve"+': '+learner_name)
    plt.grid(True)
    plt.savefig('2Chart_nn_VC1.png')     # save plot                      # <<
    # plt.show()
    plt.close()


    ###################################################################################
    ###################################################################################
    # Vailidation curve/ Hyper parameter tuning #2
    vc2_name = 'Initial Learning Rate'                               # <<
    param="learning_rate_init"                                    # <<
    param_range = [0.000001, 0.00001, 0.0001, 0.001, 0.002, 0.003, 0.005, 0.01, 0.05, 0.1]# 0.15, 0.2, 0.25, 0.3, 0.4]  # <<
    scoring_metric = 'recall'  # // accuracy, precision
    train_scores_vc2, test_scores_vc2 = \
        validation_curve(estimator=MLPClassifier(),                # <<
                        X=X_train,
                        y=y_train,
                        param_name=param,
                        param_range=param_range,
                        cv=4,
                        scoring=scoring_metric,
                        n_jobs=2,)


    train_scores_mean_vc2 = np.mean(train_scores_vc2, axis=1)
    test_scores_mean_vc2 = np.mean(test_scores_vc2, axis=1)
    df = pd.DataFrame({vc2_name: param_range, 'Training Score': train_scores_mean_vc2, 'Testing Score': test_scores_mean_vc2})


    df.plot(x=0, y=[1,2], kind='line',logx=True) # Positions of cols      # <<
    plt.ylabel(scoring_metric.capitalize()) #("Accuracy")
    plt.suptitle("Validation Curve"+': '+learner_name)
    plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.7f}'))  # 3 decimal places
    plt.grid(True)
    plt.savefig('2Chart_nn_VC2.png')     # save plot                      # <<
    # plt.show()
    plt.close()


    ###################################################################################
    ###################################################################################
    ###################################################################################
    # Vailidation curve/ Hyper parameter tuning #3     # Loss Curve
    vc3_name = 'Iterations'  # <<
    param = "max_iter"  # <<
    param_range = range(25, 500, 50)  # <<
    scoring_metric = 'neg_log_loss'  # // accuracy, precision
    train_scores_vc3, test_scores_vc3 = \
        validation_curve(estimator=MLPClassifier(),  # <<
                         X=X_train,
                         y=y_train,
                         param_name=param,
                         param_range=param_range,
                         cv=4,
                         scoring=scoring_metric,
                         n_jobs=2, )

    train_scores_mean_vc3 = np.mean(train_scores_vc3, axis=1)
    test_scores_mean_vc3 = np.mean(test_scores_vc3, axis=1)
    df = pd.DataFrame(
        {vc3_name: param_range, 'Training Score': train_scores_mean_vc3, 'Testing Score': test_scores_mean_vc3})

    df.plot(x=0, y=[1, 2], kind='line', logx=False)  # Positions of cols      # <<
    plt.ylabel(scoring_metric.capitalize())  # ("Accuracy")
    plt.ylabel('Log loss (-ve)')
    plt.suptitle("Validation Curve" + ': ' + learner_name)
    plt.grid(True)
    plt.savefig('2Chart_nn_VC3.png')  # save plot                      # <<
    # plt.show()
    plt.close()




    ###################################################################################
    ###################################################################################
    ###################################################################################
    ###################################################################################
    # Learning Curve
    scoring_metric = 'recall'  # // accuracy, precision
    train_sizes_lc, train_scores_lc, valid_scores_lc, fit_times_lc, _ = \
        learning_curve(estimator=MLPClassifier(hidden_layer_sizes=(30,5), learning_rate_init=0.000001, max_iter=300),                # <<
                        X=X_train,
                        y=y_train,
                        scoring=scoring_metric,
                        cv=3,
                        n_jobs=2,
                        train_sizes=[ 0.15, 0.3,  0.45, 0.6,  0.8, 1],
                        # train_sizes=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                        return_times=True,)

    train_scores_mean = np.mean(train_scores_lc, axis=1)
    valid_scores_mean = np.mean(valid_scores_lc, axis=1)
    fit_times_mean = np.mean(fit_times_lc, axis=1)

    df = pd.DataFrame({'Sample Size': train_sizes_lc, 'Training Score': train_scores_mean, 'Cross-validation Score': valid_scores_mean})


    df.plot(x=0, y=[1,2]) # Positions of cols

    plt.ylabel(scoring_metric.capitalize()) #("Accuracy")
    plt.xlabel("Sample Size")
    plt.suptitle("Learning Curve"+': '+learner_name)
    plt.grid(True)
    plt.savefig('2Chart_nn_LC.png')     # save plot
    # plt.show()
    plt.close()



# ensemble.GradientBoostingClassifier(*[, ...]) Gradient Boosting for classification.
def decision_tree_boosting__(X_train, X_test, y_train, y_test):

    learner_name = "Decision Tree with Boosting"
    print(learner_name)
    learner = AdaBoostClassifier(n_estimators=55, learning_rate=0.2)           # 75/.4, 5,.2

    t1 = tm.time()
    learner.fit(X_train, y_train)  # 0. Train
    t2 = tm.time()

    t3 = tm.time()
    y_pred = learner.predict(X_test) # 4. Test (Actual test/ Out sample)
    t4 = tm.time()

    y_pred_in = learner.predict(X_train) # 6. Test (On training set itself/ In sample)


    ###########################
    # Clock Times
    dt_train_time = t2 - t1 # Train
    dt_query_time = t4 - t3 # Test/ Query
    print("Training time:", dt_train_time)
    print("Testing time:", dt_query_time)

    ###########################
    ###########################
    # Accuracy/confusion matrix
    test_accuracy=accuracy_score(y_test, y_pred)
    test_recall=recall_score(y_test, y_pred, pos_label=1) # +ve class label is "2" from {1,2} or "1" from {0,1}
    print("Accuracy",test_accuracy)
    print("Recall",test_recall)
    print('\n')




    ###################################################################################
    # Vailidation curve/ Hyper parameter tuning #1
    vc1_name = 'Weak Learners'                                       # <<
    param="n_estimators"                                             # <<
    param_range = range(5,76,5)                                      # <<
    scoring_metric = 'recall'  # // accuracy, precision
    train_scores_vc1, test_scores_vc1 = \
        validation_curve(estimator=AdaBoostClassifier(),             # <<
                        X=X_train,
                        y=y_train,
                        param_name=param,
                        param_range=param_range,
                        cv=4,
                        scoring=scoring_metric,
                        n_jobs=2,)

    train_scores_mean_vc1 = np.mean(train_scores_vc1, axis=1)
    test_scores_mean_vc1 = np.mean(test_scores_vc1, axis=1)
    df = pd.DataFrame({vc1_name: param_range, 'Training Score': train_scores_mean_vc1, 'Testing Score': test_scores_mean_vc1})


    df.plot(x=0, y=[1,2], kind='line',logx=False) # Positions of cols           # <<
    plt.xticks(param_range, param_range)
    plt.ylabel(scoring_metric.capitalize()) #("Accuracy")
    plt.suptitle("Validation Curve"+': '+learner_name)
    plt.grid(True)
    plt.savefig('2Chart_dtB_VC1.png')     # save plot                      # <<
    # plt.show()
    plt.close()


    ###################################################################################
    ###################################################################################
    # Vailidation curve/ Hyper parameter tuning #2
    vc2_name = 'Learning Rate'                               # <<
    param="learning_rate"                                    # <<
    param_range = [0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6]  # <<
    scoring_metric = 'recall'  # // accuracy, precision
    train_scores_vc2, test_scores_vc2 = \
        validation_curve(estimator=AdaBoostClassifier(),     # <<
                        X=X_train,
                        y=y_train,
                        param_name=param,
                        param_range=param_range,
                        cv=4,
                        scoring=scoring_metric,
                        n_jobs=2,)


    train_scores_mean_vc2 = np.mean(train_scores_vc2, axis=1)
    test_scores_mean_vc2 = np.mean(test_scores_vc2, axis=1)
    df = pd.DataFrame({vc2_name: param_range, 'Training Score': train_scores_mean_vc2, 'Testing Score': test_scores_mean_vc2})


    df.plot(x=0, y=[1,2], kind='line',logx=True) # Positions of cols      # <<
    plt.xticks(param_range, param_range)
    plt.ylabel(scoring_metric.capitalize()) #("Accuracy")
    plt.suptitle("Validation Curve"+': '+learner_name)
    plt.grid(True)
    plt.savefig('2Chart_dtB_VC2.png')     # save plot                      # <<
    # plt.show()
    plt.close()




    ###################################################################################
    ###################################################################################
    ###################################################################################
    # Learning Curve
    scoring_metric = 'recall'  # // accuracy, precision
    train_sizes_lc, train_scores_lc, valid_scores_lc, fit_times_lc, _ = \
        learning_curve(estimator=AdaBoostClassifier(n_estimators=55, learning_rate=0.2),
                        X=X_train,
                        y=y_train,
                        scoring=scoring_metric,
                        cv=4,
                        n_jobs=2,
                        train_sizes=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                        return_times=True,)

    train_scores_mean = np.mean(train_scores_lc, axis=1)
    valid_scores_mean = np.mean(valid_scores_lc, axis=1)
    fit_times_mean = np.mean(fit_times_lc, axis=1)
    df = pd.DataFrame({'Sample Size': train_sizes_lc, 'Training Score': train_scores_mean, 'Cross-validation Score': valid_scores_mean})


    df.plot(x=0, y=[1,2]) # Positions of cols
    plt.ylabel(scoring_metric.capitalize()) #("Accuracy")
    plt.xlabel("Sample Size")
    plt.suptitle("Learning Curve"+': '+learner_name)
    plt.grid(True)
    plt.savefig('2Chart_dtB_LC.png')     # save plot
    # plt.show()
    plt.close()


def decision_tree_boosting(X_train, X_test, y_train, y_test):

    learner_name = "Decision Tree with Boosting"
    print(learner_name)
    learner = GradientBoostingClassifier(n_estimators=55, learning_rate=0.2,ccp_alpha=0.001)           # 75/.4, 5,.2

    t1 = tm.time()
    learner.fit(X_train, y_train)  # 0. Train
    t2 = tm.time()

    t3 = tm.time()
    y_pred = learner.predict(X_test) # 4. Test (Actual test/ Out sample)
    t4 = tm.time()

    y_pred_in = learner.predict(X_train) # 6. Test (On training set itself/ In sample)


    ###########################
    # Clock Times
    dt_train_time = t2 - t1 # Train
    dt_query_time = t4 - t3 # Test/ Query
    print("Training time:", dt_train_time)
    print("Testing time:", dt_query_time)

    ###########################
    ###########################
    # Accuracy/confusion matrix
    test_accuracy=accuracy_score(y_test, y_pred)
    test_recall=recall_score(y_test, y_pred, pos_label=1) # +ve class label is "2" from {1,2} or "1" from {0,1}
    print("Accuracy",test_accuracy)
    print("Recall",test_recall)
    print('\n')




    ###################################################################################
    # Vailidation curve/ Hyper parameter tuning #1
    vc1_name = 'Weak Learners'                                       # <<
    param="n_estimators"                                             # <<
    param_range = range(5,200,15)                                      # <<
    scoring_metric = 'recall'  # // accuracy, precision
    train_scores_vc1, test_scores_vc1 = \
        validation_curve(estimator=GradientBoostingClassifier(),             # <<
                        X=X_train,
                        y=y_train,
                        param_name=param,
                        param_range=param_range,
                        cv=4,
                        scoring=scoring_metric,
                        n_jobs=2,)

    train_scores_mean_vc1 = np.mean(train_scores_vc1, axis=1)
    test_scores_mean_vc1 = np.mean(test_scores_vc1, axis=1)
    df = pd.DataFrame({vc1_name: param_range, 'Training Score': train_scores_mean_vc1, 'Testing Score': test_scores_mean_vc1})


    df.plot(x=0, y=[1,2], kind='line',logx=False) # Positions of cols           # <<
    plt.xticks(param_range, param_range)
    plt.ylabel(scoring_metric.capitalize()) #("Accuracy")
    plt.suptitle("Validation Curve"+': '+learner_name)
    plt.grid(True)
    plt.savefig('2Chart_dtB_VC1.png')     # save plot                      # <<
    # plt.show()
    plt.close()


    ###################################################################################
    ###################################################################################
    # Vailidation curve/ Hyper parameter tuning #2
    vc2_name = 'Learning Rate'                               # <<
    param="learning_rate"                                    # <<
    param_range = [0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6]  # <<
    scoring_metric = 'recall'  # // accuracy, precision
    train_scores_vc2, test_scores_vc2 = \
        validation_curve(estimator=GradientBoostingClassifier(),     # <<
                        X=X_train,
                        y=y_train,
                        param_name=param,
                        param_range=param_range,
                        cv=4,
                        scoring=scoring_metric,
                        n_jobs=2,)


    train_scores_mean_vc2 = np.mean(train_scores_vc2, axis=1)
    test_scores_mean_vc2 = np.mean(test_scores_vc2, axis=1)
    df = pd.DataFrame({vc2_name: param_range, 'Training Score': train_scores_mean_vc2, 'Testing Score': test_scores_mean_vc2})


    df.plot(x=0, y=[1,2], kind='line',logx=True) # Positions of cols      # <<
    plt.xticks(param_range, param_range)
    plt.ylabel(scoring_metric.capitalize()) #("Accuracy")
    plt.suptitle("Validation Curve"+': '+learner_name)
    plt.grid(True)
    plt.savefig('2Chart_dtB_VC2.png')     # save plot                      # <<
    # plt.show()
    plt.close()




    ###################################################################################
    ###################################################################################
    ###################################################################################
    # Learning Curve
    scoring_metric = 'recall'  # // accuracy, precision
    train_sizes_lc, train_scores_lc, valid_scores_lc, fit_times_lc, _ = \
        learning_curve(estimator=GradientBoostingClassifier(n_estimators=55, learning_rate=0.2,ccp_alpha=0.001),
                        X=X_train,
                        y=y_train,
                        scoring=scoring_metric,
                        cv=4,
                        n_jobs=2,
                        train_sizes=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                        return_times=True,)

    train_scores_mean = np.mean(train_scores_lc, axis=1)
    valid_scores_mean = np.mean(valid_scores_lc, axis=1)
    fit_times_mean = np.mean(fit_times_lc, axis=1)
    df = pd.DataFrame({'Sample Size': train_sizes_lc, 'Training Score': train_scores_mean, 'Cross-validation Score': valid_scores_mean})


    df.plot(x=0, y=[1,2]) # Positions of cols
    plt.ylabel(scoring_metric.capitalize()) #("Accuracy")
    plt.xlabel("Sample Size")
    plt.suptitle("Learning Curve"+': '+learner_name)
    plt.grid(True)
    plt.savefig('2Chart_dtB_LC.png')     # save plot
    # plt.show()
    plt.close()



def support_vector_machines(X_train, X_test, y_train, y_test):

    learner_name = "Support Vector Machines"
    print(learner_name)
    learner = svm.SVC(C=7.5,kernel='poly')  #9.5poly
    # learner = svm.SVC(C=5.5,kernel='linear')

    t1 = tm.time()
    learner.fit(X_train, y_train)  # 0. Train
    t2 = tm.time()

    t3 = tm.time()
    y_pred = learner.predict(X_test) # 4. Test (Actual test/ Out sample)
    t4 = tm.time()

    y_pred_in = learner.predict(X_train) # 6. Test (On training set itself/ In sample)


    ###########################
    # Clock Times
    dt_train_time = t2 - t1 # Train
    dt_query_time = t4 - t3 # Test/ Query
    print("Training time:", dt_train_time)
    print("Testing time:", dt_query_time)

    ###########################
    ###########################
    # Accuracy/confusion matrix
    test_accuracy=accuracy_score(y_test, y_pred)
    test_recall=recall_score(y_test, y_pred, pos_label=1) # +ve class label is "2" from {1,2} or "1" from {0,1}
    print("Accuracy",test_accuracy)
    print("Recall",test_recall)
    print('\n')




    ###################################################################################
    # Vailidation curve/ Hyper parameter tuning #1
    vc1_name = 'Regularization parameter'                      # <<
    param="C"                                                   # <<
    param_range = np.arange(5,250,25)                             # <<
    scoring_metric = 'recall'  # // accuracy, precision
    train_scores_vc1, test_scores_vc1 = \
        validation_curve(estimator=svm.SVC(),                # <<
                        X=X_train,
                        y=y_train,
                        param_name=param,
                        param_range=param_range,
                        cv=4,
                        scoring=scoring_metric,
                        n_jobs=2,)

    train_scores_mean_vc1 = np.mean(train_scores_vc1, axis=1)
    test_scores_mean_vc1 = np.mean(test_scores_vc1, axis=1)
    df = pd.DataFrame({vc1_name: param_range, 'Training Score': train_scores_mean_vc1, 'Testing Score': test_scores_mean_vc1})


    df.plot(x=0, y=[1,2], kind='line',logx=False) # Positions of cols           # <<
    plt.xticks(param_range, param_range)
    plt.ylabel(scoring_metric.capitalize()) #("Accuracy")
    plt.suptitle("Validation Curve"+': '+learner_name)
    plt.grid(True)
    plt.savefig('2Chart_svm_VC1.png')     # save plot                      # <<
    # plt.show()
    plt.close()


    ###################################################################################
    ###################################################################################
    # Vailidation curve/ Hyper parameter tuning #2
    vc2_name = 'Kernel Type'                               # <<
    param="kernel"                                    # <<
    param_range = ['linear', 'poly','rbf',]  # <<
    scoring_metric = 'recall'  # // accuracy, precision
    train_scores_vc2, test_scores_vc2 = \
        validation_curve(estimator=svm.SVC(gamma='auto'),     # <<
                        X=X_train,
                        y=y_train,
                        param_name=param,
                        param_range=param_range,
                        cv=4,
                        scoring=scoring_metric,
                        n_jobs=2,)


    train_scores_mean_vc2 = np.mean(train_scores_vc2, axis=1)
    test_scores_mean_vc2 = np.mean(test_scores_vc2, axis=1)
    df = pd.DataFrame({vc2_name: param_range, 'Training Score': train_scores_mean_vc2, 'Testing Score': test_scores_mean_vc2})


    df.plot(x=0, y=[1,2], kind='bar',logx=False) # Positions of cols      # <<
    plt.xticks(rotation='horizontal')
    plt.ylabel(scoring_metric.capitalize()) #("Accuracy")
    plt.suptitle("Validation Curve"+': '+learner_name)
    plt.grid(True)
    plt.savefig('2Chart_svm_VC2.png')     # save plot                      # <<
    # plt.show()
    plt.close()




    ###################################################################################
    ###################################################################################
    ###################################################################################
    # Learning Curve
    scoring_metric = 'recall'  # // accuracy, precision
    train_sizes_lc, train_scores_lc, valid_scores_lc, fit_times_lc, _ = \
        learning_curve(estimator=svm.SVC(C=7.5,kernel='poly'),
                        X=X_train,
                        y=y_train,
                        scoring=scoring_metric,
                        cv=4,
                        n_jobs=2,
                        train_sizes=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                        return_times=True,)

    train_scores_mean = np.mean(train_scores_lc, axis=1)
    valid_scores_mean = np.mean(valid_scores_lc, axis=1)
    fit_times_mean = np.mean(fit_times_lc, axis=1)
    df = pd.DataFrame({'Sample Size': train_sizes_lc, 'Training Score': train_scores_mean, 'Cross-validation Score': valid_scores_mean})


    df.plot(x=0, y=[1,2]) # Positions of cols
    plt.ylabel(scoring_metric.capitalize()) #("Accuracy")
    plt.xlabel("Sample Size")
    plt.suptitle("Learning Curve"+': '+learner_name)
    plt.grid(True)
    plt.savefig('2Chart_svm_LC.png')     # save plot
    # plt.show()
    plt.close()



def k_nearest_neighbors(X_train, X_test, y_train, y_test):

    learner_name = "K-Nearest Neighbors"
    print(learner_name)
    learner = KNeighborsClassifier(n_neighbors=8,weights='distance')

    t1 = tm.time()
    learner.fit(X_train, y_train)  # 0. Train
    t2 = tm.time()


    t3 = tm.time()
    y_pred = learner.predict(X_test) # 4. Test (Actual test/ Out sample)
    t4 = tm.time()

    y_pred_in = learner.predict(X_train) # 6. Test (On training set itself/ In sample)


    ###########################
    # Clock Times
    dt_train_time = t2 - t1 # Train
    dt_query_time = t4 - t3 # Test/ Query
    print("Training time:", dt_train_time)
    print("Testing time:", dt_query_time)

    ###########################
    ###########################
    # Accuracy/confusion matrix
    test_accuracy=accuracy_score(y_test, y_pred)
    test_recall=recall_score(y_test, y_pred, pos_label=1) # +ve class label is "2" from {1,2} or "1" from {0,1}
    print("Accuracy",test_accuracy)
    print("Recall",test_recall)
    print('\n')


    ###################################################################################
    # Vailidation curve/ Hyper parameter tuning #1
    vc1_name = 'Number of neighbors K'                             # <<
    param="n_neighbors"                                            # <<
    param_range = range(2,20,2)                                      # <<
    scoring_metric = 'recall'  # // accuracy, precision
    train_scores_vc1, test_scores_vc1 = \
        validation_curve(estimator=KNeighborsClassifier(),               # <<
                        X=X_train,
                        y=y_train,
                        param_name=param,
                        param_range=param_range,
                        cv=4,
                        scoring=scoring_metric,
                        n_jobs=2,)

    train_scores_mean_vc1 = np.mean(train_scores_vc1, axis=1)
    test_scores_mean_vc1 = np.mean(test_scores_vc1, axis=1)
    df = pd.DataFrame({vc1_name: param_range, 'Training Score': train_scores_mean_vc1, 'Testing Score': test_scores_mean_vc1})


    df.plot(x=0, y=[1,2], kind='line',logx=False) # Positions of cols         # <<
    plt.xticks(param_range, param_range)
    plt.ylabel(scoring_metric.capitalize()) #("Accuracy")
    plt.suptitle("Validation Curve"+': '+learner_name)
    plt.grid(True)
    plt.savefig('2Chart_knn_VC1.png')     # save plot                        # <<
    # plt.show()
    plt.close()


    ###################################################################################
    ###################################################################################
    # Vailidation curve/ Hyper parameter tuning #2
    vc2_name = 'Weights'                               # <<
    param='weights'                                   # <<
    param_range = ['uniform','distance']        # <<
    scoring_metric = 'recall'  # // accuracy, precision
    train_scores_vc2, test_scores_vc2 = \
        validation_curve(estimator=KNeighborsClassifier(),     # <<
                        X=X_train,
                        y=y_train,
                        param_name=param,
                        param_range=param_range,
                        cv=4,
                        scoring=scoring_metric,
                        n_jobs=2,)


    train_scores_mean_vc2 = np.mean(train_scores_vc2, axis=1)
    test_scores_mean_vc2 = np.mean(test_scores_vc2, axis=1)
    df = pd.DataFrame({vc2_name: param_range, 'Training Score': train_scores_mean_vc2, 'Testing Score': test_scores_mean_vc2})


    df.plot(x=0, y=[1,2], kind='bar',logx=False) # Positions of cols      # <<
    plt.xticks(rotation='horizontal')
    plt.ylabel(scoring_metric.capitalize()) #("Accuracy")
    plt.suptitle("Validation Curve"+': '+learner_name)
    plt.grid(True)
    plt.savefig('2Chart_knn_VC2.png')     # save plot                      # <<
    # plt.show()
    plt.close()




    ###################################################################################
    ###################################################################################
    ###################################################################################
    # Learning Curve
    scoring_metric = 'recall'  # // accuracy, precision
    train_sizes_lc, train_scores_lc, valid_scores_lc, fit_times_lc, _ = \
        learning_curve(estimator=KNeighborsClassifier(n_neighbors=8,weights='distance'),                      # <<
                        X=X_train,
                        y=y_train,
                        scoring=scoring_metric,
                        cv=4,
                        n_jobs=2,
                        train_sizes=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                        return_times=True,)


    train_scores_mean = np.mean(train_scores_lc, axis=1)
    valid_scores_mean = np.mean(valid_scores_lc, axis=1)
    fit_times_mean = np.mean(fit_times_lc, axis=1)
    df = pd.DataFrame({'Sample Size': train_sizes_lc, 'Training Score': train_scores_mean, 'Cross-validation Score': valid_scores_mean})



    df.plot(x=0, y=[1,2]) # Positions of cols
    plt.ylabel(scoring_metric.capitalize()) #("Accuracy")
    plt.xlabel("Sample Size")
    plt.suptitle("Learning Curve"+': '+learner_name)
    plt.grid(True)
    plt.savefig('2Chart_knn_LC.png')     # save plot                      # <<
    # plt.show()
    plt.close()

def main():


    ################################################################################
    # Data set #2
    ################################################################################
    # Age	year	nodes	Survival
    cancer_df = pd.read_csv(path.join('data','haberman.data'))
    # cancer_df.rename(columns={'Class':'Class_category'}, inplace=True)
    dataset = 2

    cancer_df.columns = ['Age',	'Year',	'Nodes','survival']
    # y = cancer_df.iloc[:, -1]


    # Negative class/Long survival :0 , Positive class/short life :1
    cancer_df.survival.replace((1, 2), (0, 1), inplace=True)


    y = cancer_df['survival']
    X = cancer_df.drop(['survival'], axis=1)


    # define oversampling strategy
    oversample = RandomOverSampler(sampling_strategy='minority')
    # fit and apply the transform
    X_over, y_over = oversample.fit_resample(X, y)
    X = X_over
    y = y_over

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3240)



    df_count = pd.value_counts(y)
    df_count.plot( kind='bar',color=['green', 'blue']) # Positions of cols      # <<

    plt.suptitle('Data Distribution')
    plt.xlabel('Class')
    plt.ylabel('Frequency')
    plt.xticks(rotation='horizontal')
    # if dataset ==1:
    #     plt.xticks([0,1],['Kecimen','Besni'])
    #     plt.savefig('Raisin_Balance.png')  # save plot
    # else:
    plt.xticks([0,1],['Survived Longer','Died Sooner'])
    plt.savefig('Cancer_Balance.png')  # save plot

    # plt.show()
    plt.close()






    decision_tree(X_train, X_test, y_train, y_test)
    neural_network(X_train, X_test, y_train, y_test)
    decision_tree_boosting(X_train, X_test, y_train, y_test)
    support_vector_machines(X_train, X_test, y_train, y_test)
    k_nearest_neighbors(X_train, X_test, y_train, y_test)


    print()

if __name__ == "__main__":
    main()

