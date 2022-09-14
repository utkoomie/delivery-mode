#------------------------------------------------------------------
# Machine-learning support utilities for analysis using CDC data.
# 
# Copyright 2018-2022 Karl W. Schulz
# 
# Dell Medical School, University of Texas
#------------------------------------------------------------------
from sklearn.metrics import accuracy_score, f1_score, brier_score_loss
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from sklearn.utils import column_or_1d
from sklearn.model_selection import cross_validate
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import utils
import pandas
import shap

#--
# Summarize trained model scorces
def score_printout(X_test, y_test, fittedModel):
    print ("AUC-ROC model score   = %f" % roc_auc_score(y_test, fittedModel.predict_proba(X_test)[:,1]))  
#    print ("AUC-ROC model score   = %f" % roc_auc_score  (y_test, fittedModel.predict(X_test)))
    print ("Accuracy model score  = %f" % accuracy_score (y_test, fittedModel.predict(X_test)))
    print ("Recall model score    = %f" % recall_score(y_test, fittedModel.predict(X_test)))
    print ("F1 model score        = %f" % f1_score(y_test, fittedModel.predict(X_test)))
    print ("Brier loss score      = %f" % brier_score_loss(y_test, fittedModel.predict_proba(X_test)[:,1])) 
    print ("\n")


def plot_confusion_matrix(cnfMatrix,normalized=True,title=False,plotName=None,axis=None,cbar=True):
    
    fmtType='d'
    mapColor="Blues"

    if normalized:
        cnfMatrix = cnfMatrix.astype('float') / cnfMatrix.sum(axis=1)[:, np.newaxis]
        fmtType='.1%'

    if axis is None:
        map = sns.heatmap(cnfMatrix, annot=True,fmt=fmtType,cmap=mapColor,vmin=0,vmax=1,cbar=cbar)
    else:
        map = sns.heatmap(cnfMatrix, annot=True,fmt=fmtType,cmap=mapColor,vmin=0,vmax=1,ax=axis,cbar=cbar)

    plt.gca()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    
    if title:
        plt.title(title,fontsize='8')


def calibration_curve(y_true, y_prob, normalize=False, n_bins=5,mybins=None,maxProb=1.0,
                      logFile=None,minSamples=None):
    """Compute true and predicted probabilities for a calibration curve.
     The method assumes the inputs come from a binary classifier.
     Calibration curves may also be referred to as reliability diagrams.
    Read more in the :ref:`User Guide <calibration>`.
    Parameters
    ----------
    y_true : array, shape (n_samples,)
        True targets.
    y_prob : array, shape (n_samples,)
        Probabilities of the positive class.
    normalize : bool, optional, default=False
        Whether y_prob needs to be normalized into the bin [0, 1], i.e. is not
        a proper probability. If True, the smallest value in y_prob is mapped
        onto 0 and the largest one onto 1.
    n_bins : int
        Number of bins. A bigger number requires more data.
    Returns
    -------
    prob_true : array, shape (n_bins,)
        The true probability in each bin (fraction of positives).
    prob_pred : array, shape (n_bins,)
        The mean predicted probability in each bin.
    References
    ----------
    Alexandru Niculescu-Mizil and Rich Caruana (2005) Predicting Good
    Probabilities With Supervised Learning, in Proceedings of the 22nd
    International Conference on Machine Learning (ICML).
    See section 4 (Qualitative Analysis of Predictions).
    """
    y_true = column_or_1d(y_true)
    y_prob = column_or_1d(y_prob)

    if normalize:  # Normalize predicted values into interval [0, 1]
        y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min())
    elif y_prob.min() < 0 or y_prob.max() > 1:
        raise ValueError("y_prob has values outside [0, 1] and normalize is "
                         "set to False.")

# missing with new scikit update (7.25.19 - koomie)
###    y_true = _check_binary_probabilistic_predictions(y_true, y_prob)

    #bins = np.linspace(0., 1. + 1e-8, n_bins + 1)
    bins  = np.linspace(0., maxProb + 1e-8, n_bins + 1)
    binids = np.digitize(y_prob, bins) - 1

    if minSamples is not None:
        print("minSample check:")
        for iter in range(n_bins - 2):
            # check if a bin does not have sufficient sample. If so, alter bin limits
            minlength=len(bins)-1
            bin_total = np.bincount(binids, minlength=minlength)
            if bin_total[-1] < minSamples:
                print("Not enough samples in range (%.1f to %.1f): %i " % (bins[-2],bins[-1],bin_total[-1]))
                bins = np.delete(bins,len(bins)-2)
                binids = np.digitize(y_prob, bins) - 1
            else:
                print("OK: met minSample threshold after altering last bin limit (%i)" % bin_total[-1])
                print(bins)
                break

    minlength=len(bins)         # original
    minlength=len(bins)-1       # koomie mod (8/2/18)

    bin_sums  = np.bincount(binids, weights=y_prob, minlength=minlength)
    bin_true  = np.bincount(binids, weights=y_true, minlength=minlength)
    bin_total = np.bincount(binids, minlength=minlength)

    # koomie mod (8/2/18): leave empty bins as is
    nonzero = bin_total != 0
#    prob_true = (bin_true[nonzero] / bin_total[nonzero])
#    prob_pred = (bin_sums[nonzero] / bin_total[nonzero])

    with np.errstate(divide='ignore',invalid='ignore'):
        prob_true = (bin_true / bin_total)
        prob_pred = (bin_sums / bin_total)

    if logFile is not None:
        logFile.write('model probability bin ranges and totals vs. (true prob):\n')
        for index in range (1,len(bins)):
            logFile.write('%4.2f (%4.2f,%4.2f) = %15i   (%4.2f)\n' % 
                          (prob_pred[index-1],bins[index-1],
                           bins[index],bin_total[index-1],prob_true[index-1]) )
        logFile.flush()

    zeros = np.where(bin_total == 0)[0]
    if len(zeros) > 0:
        print("Warning: no classifier probabilities found in histogram for the following bin endpoints:")
        for prob in zeros:
            print("--> [%4.2f to %4.2f]"  % (bins[prob],bins[prob+1]))

    return prob_true, prob_pred

def plot_confusion_and_reliability(cnf_matrix,mean_pred,actual_pred,filename=None,aspect=0.8,title=None):
    plt.clf()
    #subplot=utils.subplot(nrows=1,ncols=2)
    subplot=utils.subplot(nrows=2,ncols=1)

    subplot.increment(aspect=aspect)
    plot_confusion_matrix(cnf_matrix,cbar=False)

#    plt.title('Confusion Matrix',fontsize='10')
    subplot.increment(aspect='0.8')
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    plt.xlim(0,1); plt.ylim(0,1)
    plt.plot(mean_pred, actual_pred, "s-") ; plt.grid()
    plt.ylabel("Fraction of positives") ; plt.xlabel("Mean probability")
    plt.title('Reliability Curve',fontsize='10')

    # ok for horizaontal
    #plt.subplots_adjust(wspace=0.36, hspace=0.2)

    # ok for vertical
    plt.subplots_adjust(wspace=0.3, hspace=0.65)

    if title is not None:
        # ok for horizintal
        #plt.gcf().suptitle(title,y=0.85,fontsize='10')
        # ok for veritcal
        plt.gcf().suptitle(title,fontsize='10')
        plt.gcf().subplots_adjust(bottom=0.12)
        plt.gcf().subplots_adjust(top=0.85)

#    plt.tight_layout()

    if filename is not None:
        plt.savefig(filename)
        print("results saved to %s" % filename)


#--
# Run cross-validation with supplied classifier and data. First run
# with all parameters to ascertain feature importance. Then run with a
# subset of sorted featres as dictated by values in subFeatureCounts.
def cross_validation_with_subfeatures(clf,subFeatureCounts,model,params,metrics,data,modelY,weight,cv=10,
                                      feature_log=False):
    algoName = clf.__class__.__name__
    print("\nRunning %s for %s params" % (algoName,model))

    timer = utils.timer()
    
    scores = {}

    # cull out current modeling params
    modelX = data.filter(items=params[model])
    print("max # of params = %i" % modelX.shape[1])

    # First, run a fit for all params and examine importance
    timer.start()
    clf.fit(modelX, modelY.values.ravel(),weight)
    timer.stop("to perform single train using all parameters")

    classifierType = type(clf).__name__

    if feature_log:
        pos_class_prob_sorted = clf.feature_log_prob_[1, :].argsort()
        features = pandas.Series(data=None,index=np.take(modelX.columns, pos_class_prob_sorted))
    else:
        if (classifierType == 'XGBClassifier'):
            # feature importance based on SHAP values for classifiers that support it
            print('Sorting features based on SHAP importance')
            explainer   = shap.TreeExplainer(clf)
            shap_values = explainer.shap_values(modelX)
            features    = pandas.Series(np.absolute(shap_values).mean(axis=0),index=modelX.columns).sort_values(ascending=False)
        else:
            # feature importance based on gini
            print('Sorting features based on gini importance')
            features = pandas.Series(clf.feature_importances_,index=modelX.columns).sort_values(ascending=False)

    # Next, do cross-validation with various parameter counts
    for numFeatures in subFeatureCounts:
        print("\n--> Running cross-validation for %i features" % numFeatures)
    
        label = algoName + '-' + model + '-' + str(numFeatures)
        modelX = data.filter(items=features.index[:numFeatures])

        timer.start()
        scores[label] = cross_validate(clf, modelX, modelY.values.ravel(), cv=cv,
                                       verbose=0,scoring=metrics,fit_params={'sample_weight':weight})
        timer.stop("to perform cross-validation for %i features" % numFeatures)
    
    # Lastly, cross-validation with all parameters
    modelX = data.filter(items=params[model])
    numFeatures = modelX.shape[1]

    print("\n--> Running cross-validation for %i features" % numFeatures)
    label = algoName + '-' + model + '-' + str(modelX.shape[1])
    timer.start()
    scores[label] = cross_validate(clf, modelX, modelY.values.ravel(), n_jobs=1,cv=10,
                                   verbose=0,scoring=metrics,fit_params={'sample_weight':weight})
    timer.stop("to perform cross-validation for all %i features" % numFeatures)
    
    print("\n%24s  %13s  %13s   %13s   %13s  %13s" % ('Model','AUC'.center(13),'Accuracy'.center(13),
                                                      'Recall'.center(13),'F1'.center(13),'Brier'.center(13)))
    print("-" * 110)
    for algo, results in scores.items():
        auc,auc_ci     = results['test_roc_auc'         ].mean(), results['test_roc_auc'         ].std() * 2
        acc,acc_ci     = results['test_accuracy'        ].mean(), results['test_accuracy'        ].std() * 2
        rec,rec_ci     = results['test_recall'          ].mean(), results['test_recall'          ].std() * 2
        f1,f1_ci       = results['test_f1'              ].mean(), results['test_f1'              ].std() * 2
        brier,brier_ci = results['test_brier_score_loss'].mean(), results['test_brier_score_loss'].std() * 2

        # note: cross-validation scoring is always returned such that
        # larger values are better (so a negative sign is included for
        # Brier score).  

        brier = abs(brier)

        algoShort = algo.replace('Classifier','')
   
        print("%24s  %.03f +/- %0.3f  %.03f +/- %0.3f  %.03f +/- %0.3f  %.03f +/- %0.3f  %.03f +/- %0.3f" % 
              (algoShort,auc,auc_ci,acc,acc_ci,rec,rec_ci,f1,f1_ci,brier,brier_ci))

    return scores
