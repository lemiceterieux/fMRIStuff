import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
import nilearn
import nilearn.decoding
from sklearn.model_selection import KFold
import sys
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import LinearSVC
from sklearn.gaussian_process import GaussianProcessClassifier as GPC
from sklearn.decomposition import PCA
import nibabel as nib


# Define my estimator, PCA preprocessing with standard scalar LinearSVC
class PCAPreProcSVC(BaseEstimator, ClassifierMixin):
    def __init__(self, demo_param='', n_comps=.9):
        self.demo_param = demo_param
        self.n_comps = n_comps

    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        self.X_ = X
        self.y_ = y
        self.clf = LinearSVC()
        self.pca = PCA(self.n_comps)
        self.scaler = StandardScaler()        
        self.clf.fit(self.pca.fit_transform(self.scaler.fit_transform(X)), y)
        # Return the classifier
        return self

    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self)
        
        # Input validation
        X = check_array(X)
        
        return self.clf.predict(self.pca.transform(self.scaler.transform(X))) 

nR= 13
diam = 6
radius = diam/2

Runs = np.load("Runs.npy")
affine = np.load("affine.npy")
Rnifti = nib.Nifti1Image(Runs.reshape(nR,*Runs.shape[1:4],2).transpose(0,4,1,2,3).reshape(nR*2, *Runs.shape[1:4]).transpose(1,2,3,0), affine=affine)

# Anatomical MAsk
anatMask = nib.Nifti1Image((nib.load("rnewT1_2.nii").get_fdata() > 60).astype(int),affine=affine)
# Define which Runs belong to the same session
sessions = np.array([np.arange(nR) for i in range(2)]).reshape(-1)
# Labels of the Runs
targets = np.array(["Claim" if i > nR else "NoClaim" for i in range(26)])

# Leave One Run Out is an Instantiation of K=nR
cv = KFold(n_splits=nR)


# Define Searchlight parameters, paralellizing over the number of Runs
sl = nilearn.decoding.SearchLight(anatMask,process_mask_img=anatMask, radius=radius,n_jobs=nR,cv=cv, verbose=1, estimator=PCAPreProcSVC())
print("Starting Searchlight!")
# Fit the searchlights
sl_map = sl.fit(Rnifti, targets, sessions)
# Save the scores
slScores = nib.Nifti1Image(sl_map.scores_, affine=Rnifti.affine)
nib.save(slScores,"PCASVCNILearn.nii")
